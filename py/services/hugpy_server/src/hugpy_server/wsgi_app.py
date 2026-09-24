"""hugpy server: the Flask composition root.

``get_hugpy_flask()`` / ``create_app()`` build the app: every route
blueprint, the /api dual mounts, the operator/member/video gates, the built
React console (``console_dist`` package data) and — through
``hugpy_server.wiring.install_all`` — every cross-package seam listed in
``py/WIRING.md``. ``main()`` is the ``hugpy-serve`` console script (gunicorn
on POSIX, waitress on Windows, the Flask dev server as the last resort).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from importlib import resources as _resources

from flask import Response, redirect, send_from_directory

from abstract_flask import get_Flask_app

from hugpy_server.app import routes as routes
from hugpy_server.app.routes.worker_routes import worker_bp
from hugpy_server.app.routes.phone_brick_routes import phone_brick_bp
from hugpy_server.app.routes.discord_routes import discord_bp
from hugpy_server.app.routes.video_routes import video_bp
from hugpy_server.app.routes.agent_routes import agent_bp
from hugpy_server.app.routes.messages_routes import messages_bp

logger = logging.getLogger(__name__)

# Video Intelligence worker daemon is started ONCE per process at app init
# (guarded below). Module-level so re-entrant app creation can't spawn a second.
_VIDEO_DAEMON_STARTED = False
_ADMISSION_RUNNER_STARTED = False


class ApiPrefixMiddleware:
    """Strip a leading /api from the request path, exactly like the public
    nginx does. With it, the app standalone (no proxy in front) accepts both
    the bare paths gunicorn has always served (/health, /v1/...) and the
    /api-prefixed paths every client uses (/api/health, /api/v1/...). This is
    what lets one process serve UI + API with no nginx at all."""

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if path == "/api" or path.startswith("/api/"):
            environ["PATH_INFO"] = path[len("/api"):] or "/"
        return self.wsgi_app(environ, start_response)


# The built React console ships as package data: console_dist/ inside the
# wheel, one directory per mount (see BUILD_CONSOLE.md):
#   /        react/ui                      -> console_dist/index.html
#   /fleet   react/agents_ui               -> console_dist/fleet/
#   /media   react/media_intelligence_ui   -> console_dist/media/
#   /video   react/video_intelligence_ui   -> console_dist/video/
CONSOLE_ARMS = ("fleet", "media", "video")


def packaged_console_dir() -> str | None:
    """``console_dist`` inside the installed package (``importlib.resources``),
    or None when the package carries no bundle."""
    try:
        root = _resources.files("hugpy_server").joinpath("console_dist")
        path = str(root)
    except Exception:  # noqa: BLE001 — a zipped/odd loader is "no bundle"
        return None
    return path if os.path.isdir(path) else None


def console_dist_dir() -> str | None:
    """Where the built UI lives, if anywhere.

    HUGPY_UI_DIST (env or .env) wins; otherwise the packaged console_dist.
    A candidate counts only when it holds an index.html. Returns None when
    there is no build — API-only mode, nothing changes."""
    explicit = os.environ.get("HUGPY_UI_DIST")
    if not explicit:
        try:
            from hugpy_platform.platform_facade import env_value
            explicit = env_value("HUGPY_UI_DIST")
        except Exception:
            explicit = None
    candidates = [explicit] if explicit else []
    packaged = packaged_console_dir()
    if packaged:
        candidates.append(packaged)
    for cand in candidates:
        if cand and os.path.isfile(os.path.join(cand, "index.html")):
            return cand
    return None


_ui_dist_dir = console_dist_dir  # historical name


# index.html marker the video arm ships for server-injected demo-media bases
# (react/video_intelligence_ui/index.html). We rewrite its `null` in place.
_MEDIA_BASE_MARKER = (
    "window.__HUGPY_MEDIA_BASE__ = window.__HUGPY_MEDIA_BASE__ ?? null;"
)
# Rewritten index bytes, keyed by index.html path — computed once per process
# (the dist is immutable for a process lifetime, same as the SPA serving style).
_VIDEO_INDEX_CACHE: dict = {}


def _demo_media_override() -> str | None:
    """The demo-media base to inject into the video arm's index.html, or None.

    HUGPY_DEMO_MEDIA_DIR set → we serve the tree ourselves at /demo-media.
    Else HUGPY_DEMO_MEDIA_BASE set → point the SPA at that base.
    Neither → None (the SPA falls back to its built-in default)."""
    try:
        from hugpy_platform.platform_facade import env_value
        from hugpy_platform.app_dirs import demo_media_base, demo_media_dir
        if demo_media_dir():
            return "/demo-media"
        if env_value("HUGPY_DEMO_MEDIA_BASE"):
            return demo_media_base()
    except Exception:
        pass
    return None


def _video_index_response(video_dir: str):
    """Serve the video arm's index.html with the demo-media base injected.

    Returns None (caller falls back to plain send_from_directory) when no
    override is configured, the marker is absent, or anything goes wrong —
    the rewrite must never break SPA serving."""
    base = _demo_media_override()
    if not base:
        return None
    index_path = os.path.join(video_dir, "index.html")
    cached = _VIDEO_INDEX_CACHE.get(index_path)
    if cached is None:
        try:
            with open(index_path, "rb") as fh:
                raw = fh.read()
        except OSError:
            return None
        needle = _MEDIA_BASE_MARKER.encode("utf-8")
        if needle not in raw:
            return None  # marker missing → serve unmodified via the normal path
        repl = _MEDIA_BASE_MARKER.replace("null", json.dumps(base)).encode("utf-8")
        cached = raw.replace(needle, repl, 1)
        _VIDEO_INDEX_CACHE[index_path] = cached
    return Response(cached, mimetype="text/html")


def mount_console(app, dist_dir: str | None) -> bool:
    """Serve the built SPA from ``dist_dir``: real files from dist, everything
    else (deep links like /login) falls back to index.html. API rules are
    explicit routes, so Werkzeug always prefers them over this converter
    catch-all. Returns False (and mounts nothing — every unmatched path is a
    plain 404) when ``dist_dir`` is None or holds no index.html."""
    if not dist_dir or not os.path.isfile(os.path.join(dist_dir, "index.html")):
        return False

    # /studio — the public name for the Studio experience, whose stations live
    # INSIDE the video arm (/video/). One redirect keeps the short URL working
    # without duplicating a build or splitting the arm; the webpack devServer has
    # the same rewrite (react/ui/webpack.config.js historyApiFallback) so dev and
    # prod agree. 302 (not 301): the destination is a routing decision we may
    # revisit, and a permanent redirect would be cached in browsers forever.
    @app.route("/studio", defaults={"sub": ""})
    @app.route("/studio/<path:sub>")
    def _hugpy_studio(sub):
        return redirect("/video/" + (sub or ""), code=302)

    @app.route("/", defaults={"asset": ""})
    @app.route("/<path:asset>")
    def _hugpy_ui(asset):
        target = os.path.join(dist_dir, asset)
        if asset and os.path.isfile(target):
            # A direct hit on the video arm's index still gets the demo-media
            # base injected (same bytes as the deep-link fallback below).
            if asset == "video/index.html":
                injected = _video_index_response(os.path.join(dist_dir, "video"))
                if injected is not None:
                    return injected
            return send_from_directory(dist_dir, asset)
        # Standalone arms (fleet, media-intelligence, video-intelligence) are
        # separate SPAs shipped at <arm>/ inside the same dist (console_dist/<arm>/*).
        # Their assets are real files (served above); their deep links (/media,
        # /video/foo) must fall back to the arm's OWN index.html, not the console
        # SPA's — mirrors the webpack devServer historyApiFallback rewrite that
        # keeps the dev path working. Guarded by isfile, so an arm that isn't
        # shipped in this dist changes nothing.
        for arm in CONSOLE_ARMS:
            if (asset == arm or asset.startswith(arm + "/")) and os.path.isfile(
                os.path.join(dist_dir, arm, "index.html")
            ):
                if arm == "video":
                    injected = _video_index_response(os.path.join(dist_dir, arm))
                    if injected is not None:
                        return injected
                return send_from_directory(os.path.join(dist_dir, arm), "index.html")
        return send_from_directory(dist_dir, "index.html")

    return True


_mount_ui = mount_console  # historical name


def _daemons_enabled(start_daemons) -> bool:
    if start_daemons is not None:
        return bool(start_daemons)
    flag = (os.environ.get("HUGPY_START_DAEMONS") or "").strip().lower()
    return flag not in ("0", "false", "no", "off")


def get_hugpy_flask(name=None, allowed_origins=None, debug=False, *,
                    start_daemons=None, wire=True):
    """Build the hugpy Flask app.

    ``start_daemons`` (default: env ``HUGPY_START_DAEMONS``, on) controls the
    per-process background workers (the video job daemon); ``wire`` runs
    ``hugpy_server.wiring.install_all`` so every cross-package seam has its
    real implementation. Tests pass ``start_daemons=False``."""
    name = name or "hugpy_flask"
    # Tighten CORS for the public deployment: an explicit allowlist beats the
    # reflect-any default on a credentialed API. Comma-separated origins in
    # HUGPY_ALLOWED_ORIGINS; unset keeps prior behavior (same-origin UI calls
    # don't use CORS, so this only constrains cross-origin browser callers).
    if allowed_origins is None:
        _ao = (os.environ.get("HUGPY_ALLOWED_ORIGINS") or "").strip()
        if _ao:
            allowed_origins = [o.strip() for o in _ao.split(",") if o.strip()]
    app = get_Flask_app(
        name=name,
        routes=routes,
        allowed_origins=allowed_origins,
        debug=debug
    )
    # Bound request bodies so /uploads (and any POST) can't be an unbounded
    # memory/disk DoS. Generous default (100 MB); override via HUGPY_MAX_UPLOAD_MB.
    try:
        _mb = float(os.environ.get("HUGPY_MAX_UPLOAD_MB", "100"))
        app.config["MAX_CONTENT_LENGTH"] = int(_mb * 1024 * 1024)
    except (TypeError, ValueError):
        pass
    # Dual-mount the worker/model routes under /api as well.
    #
    # gunicorn serves these at /llm/... and /models; the /api prefix the worker
    # uses exists ONLY because the public nginx (hugpy.ai) strips it. A worker
    # that reaches central directly — e.g. over WireGuard at http://<wg-ip>:7002,
    # bypassing nginx — would 404 on its /api/llm/... calls. Registering the
    # blueprint again under /api makes /api/llm/workers/* and /api/llm/models/*
    # resolve on gunicorn itself, so the direct (proxy-less) route works
    # identically to the nginx route. The original /llm/... mount is untouched.
    try:
        app.register_blueprint(worker_bp, url_prefix="/api", name="worker_bp_api")
    except (ValueError, AssertionError):
        # Idempotent: already mounted on this app instance.
        pass

    # Same /api dual-mount for the phone-brick pool: phones register, heartbeat,
    # and fetch seeded images by reaching gunicorn directly over the VPN.
    try:
        app.register_blueprint(phone_brick_bp, url_prefix="/api", name="phone_brick_bp_api")
    except (ValueError, AssertionError):
        pass

    # Same /api dual-mount for Discord bindings: the hugpy bot reaches central
    # directly (resolve a model for a channel/user, drain the outbox) and may
    # bypass nginx, so these must resolve on gunicorn itself too.
    try:
        app.register_blueprint(discord_bp, url_prefix="/api", name="discord_bp_api")
    except (ValueError, AssertionError):
        pass

    # Same /api dual-mount for the Video Intelligence routes. The public path is
    # dev.hugpy.ai/api/video/...; host nginx strips /api before the VM, so the
    # bare /video/... mount (auto-discovered via routes/__init__) serves the
    # proxied path, and this second mount makes direct-to-gunicorn /api/video/...
    # resolve identically (mirrors worker_bp). The original /video/... mount is
    # untouched.
    try:
        app.register_blueprint(video_bp, url_prefix="/api", name="video_bp_api")
    except (ValueError, AssertionError):
        pass

    # Same /api dual-mount for the P3.1 agent-node fleet: remote agent nodes
    # register, heartbeat and pull tasks by reaching gunicorn directly over the
    # VPN, exactly like GPU workers and phone bricks. The bare /agent/... mount
    # (auto-discovered via routes/__init__) serves the nginx-proxied path.
    try:
        app.register_blueprint(agent_bp, url_prefix="/api", name="agent_bp_api")
    except (ValueError, AssertionError):
        pass

    # Same /api dual-mount for the Anthropic Messages shim, so Claude Code / the
    # Claude Agent SDK reach /v1/messages whether they hit the nginx-proxied
    # /api/v1/messages or gunicorn directly. The bare /v1/messages mount
    # (auto-discovered via routes/__init__) serves the proxied path unchanged.
    try:
        app.register_blueprint(messages_bp, url_prefix="/api",
                               name="messages_bp_api")
    except (ValueError, AssertionError):
        pass

    # Same /api dual-mount for eviction telemetry. Workers POST their event
    # batches to /api/llm/evictions/ingest through CentralClient, which may reach
    # gunicorn directly over WireGuard (no nginx to strip the prefix); the console
    # reads /api/llm/evictions and /api/llm/evictions/stream through the same
    # prefix. The bare /llm/evictions... mount (auto-discovered via
    # routes/__init__) serves the proxied path unchanged.
    try:
        from hugpy_server.app.routes.eviction_routes import eviction_bp
        app.register_blueprint(eviction_bp, url_prefix="/api",
                               name="eviction_bp_api")
    except (ValueError, AssertionError):
        pass
    except Exception as _exc:  # noqa: BLE001 — telemetry must never break boot
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "eviction telemetry routes not mounted under /api: %s", _exc)

    # Same /api dual-mount for MODEL GROUPS, so the Models tab reads
    # /api/llm/groups on the same prefix as every other panel read. Read-only —
    # the tick WRITES go through /settings/model_groups/..., which the operator
    # gate already covers; there is deliberately no group write route to mount.
    try:
        from hugpy_server.app.routes.group_routes import group_bp
        app.register_blueprint(group_bp, url_prefix="/api",
                               name="group_bp_api")
    except (ValueError, AssertionError):
        pass
    except Exception as _exc:  # noqa: BLE001 — a read surface must not break boot
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "model-group routes not mounted under /api: %s", _exc)

    # Same /api dual-mount for the EXPLICIT priority groups, so the Models tab
    # reads /api/llm/model-groups on the same prefix as every other panel. This
    # blueprint DOES carry writes (POST/PUT/PATCH/DELETE), and they are listed
    # in operator_auth._SENSITIVE — which strips the /api prefix before matching,
    # so the gate covers both mounts identically.
    try:
        from hugpy_server.app.routes.model_group_routes import model_group_bp
        app.register_blueprint(model_group_bp, url_prefix="/api",
                               name="model_group_bp_api")
    except (ValueError, AssertionError):
        pass
    except Exception as _exc:  # noqa: BLE001 — must not break boot
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "model priority-group routes not mounted under /api: %s", _exc)

    # Same /api dual-mount for the COMPUTE-ACTIVITY METRICS reads, so the Metrics
    # tab reads /api/llm/model-metrics and /api/llm/compute-actions on the same
    # prefix as every other panel. Read-only (GET), so — like the eviction and
    # model-group reads — there is nothing in operator_auth._SENSITIVE to mount.
    # The bare /llm/... mount is auto-discovered via routes/__init__.
    try:
        from hugpy_server.app.routes.metrics_routes import metrics_bp
        app.register_blueprint(metrics_bp, url_prefix="/api",
                               name="metrics_bp_api")
    except (ValueError, AssertionError):
        pass
    except Exception as _exc:  # noqa: BLE001 — a read surface must not break boot
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "compute-metrics routes not mounted under /api: %s", _exc)

    # Optional: mount the media_intelligence HTTP bridge at /media/analyze.
    #
    # The chat's media-intelligence path (ui mediaIntelligence.ts -> tryServerBridge)
    # POSTs {file:<upload ref>, kind, source:<filename>} to /api/media/analyze. nginx
    # (and ApiPrefixMiddleware) strips the /api, so the route registered IN Flask is
    # /media/analyze. The bridge runs the media_intelligence MediaPipeline and returns
    # ONE typed DocumentIntelligence record ({ok, result:{...}}) — instead of the UI
    # orchestrating the per-task /ml/* endpoints itself.
    #
    # ADDITIVE + FAULT-TOLERANT: media_intelligence (and its [bridge] extra, Flask) is
    # an OPTIONAL dependency that is NOT part of the hugpy wheel. If it isn't installed
    # in the venv, log and skip — the app must always boot. This stays hugpy's own arm;
    # the @hugpy/console PTY boundary is untouched.
    #
    # Contract: the bridge accepts {"file": <handle>} and maps it via resolve_file —
    # the one host-specific seam. The UI's `file` is whatever POST /uploads returned,
    # which today is the absolute path under UPLOADS_HOME (UploadUtils falls back to
    # `path` when the server emits no opaque id), and may be a bare id later. _resolve_upload
    # accepts EITHER form and jails it under the storage root (same jail as
    # functions.media_extract), so this can never become an arbitrary-file-read.
    try:
        from media_intelligence.bridge import build_blueprint as _build_media_bridge

        def _resolve_upload(handle):
            if not handle:
                return None
            from hugpy_platform.constants import UPLOADS_HOME, DEFAULT_ROOT
            cand = handle if os.path.isabs(handle) else os.path.join(
                UPLOADS_HOME, os.path.basename(handle)
            )
            rp = os.path.realpath(cand)
            roots = [os.path.realpath(r) for r in (UPLOADS_HOME, DEFAULT_ROOT) if r]
            if not any(rp == root or rp.startswith(root + os.sep) for root in roots):
                return None  # outside the storage root -> refuse (no arbitrary read)
            # OWNERSHIP (2026-08-06). The jail says "inside our storage"; it never
            # said "yours". Uploads are namespaced per account
            # (UPLOADS_HOME/<namespace>/…), so a MEMBER analyzing a path under
            # UPLOADS_HOME must be inside THEIR namespace — otherwise this bridge
            # would remain a read of any other account's upload by path.
            # Scoped to members ONLY (a resolvable namespace), exactly like the
            # /video/media rule: an operator, an operator-token/open-mode caller
            # and an API-key M2M caller have no namespace and stay unrestricted,
            # as do paths outside UPLOADS_HOME (job artifacts, whose per-artifact
            # rule lives in video_routes). Refuses the handle on any error.
            try:
                from hugpy_server.app.operator_auth import principal_role, principal_username, upload_namespace
                uploads_root = os.path.realpath(UPLOADS_HOME) if UPLOADS_HOME else None
                if uploads_root and (rp == uploads_root
                                     or rp.startswith(uploads_root + os.sep)):
                    ns = (upload_namespace(principal_username())
                          if principal_role() == "member" else None)
                    if ns:
                        home = os.path.join(uploads_root, ns)
                        if not rp.startswith(home + os.sep):
                            return None
            except Exception:  # noqa: BLE001
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "upload ownership check failed — refusing handle", exc_info=True)
                return None
            return rp if os.path.isfile(rp) else None

        app.register_blueprint(_build_media_bridge(resolve_file=_resolve_upload))
    except ImportError as _exc:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "media_intelligence bridge not mounted (optional dep missing): %s", _exc
        )
    except Exception as _exc:
        import logging as _logging
        _logging.getLogger(__name__).error(
            "media_intelligence bridge install failed: %s", _exc
        )

    # Standalone (distribution) mode: accept /api/* without a proxy, and serve
    # the built UI when one exists. Both are no-ops in the proxied dev/prod
    # topology (nginx already strips /api; webpack serves the UI).
    app.wsgi_app = ApiPrefixMiddleware(app.wsgi_app)
    dist_dir = console_dist_dir()
    if not mount_console(app, dist_dir):
        logger.info("no console bundle found (API-only mode); see BUILD_CONSOLE.md")
    # Self-hosted demo media: only when HUGPY_DEMO_MEDIA_DIR is configured does
    # /demo-media/* exist at all (send_from_directory path-jails relpath, so
    # ../ traversal cannot escape the configured tree). The video arm's
    # index.html is rewritten to point at it — see _video_index_response.
    try:
        from hugpy_platform.app_dirs import demo_media_dir as _demo_media_dir
        _dm_dir = _demo_media_dir()
    except Exception:
        _dm_dir = ""
    if _dm_dir:
        @app.route("/demo-media/<path:relpath>")
        def _hugpy_demo_media(relpath):
            return send_from_directory(_dm_dir, relpath, conditional=True)
    # Abandon-on-disconnect (2026-07-27 outage): bind a probe for each request's
    # client socket so a WSGI thread blocked on model work can discover that its
    # caller left and give the request slot back — instead of holding one of the
    # site's 24 slots for the rest of a 25-minute cold hold, serving nobody.
    # Inert wherever no socket is published (waitress / dev server / no gunicorn)
    # and switchable off via HUGPY_CLIENT_DISCONNECT_ABANDON=off. Never break boot.
    try:
        from hugpy_platform import client_liveness
        client_liveness.install(app)
    except Exception as _exc:
        import logging as _logging
        _logging.getLogger(__name__).error(
            "client-liveness probe install failed: %s", _exc)

    # Server-side operator auth gate on console-side management routes. Inert
    # until HUGPY_AUTH_MODE=external (or HUGPY_OPERATOR_TOKEN is set), so this
    # is safe to deploy and verify before activation.
    try:
        from hugpy_server.app.operator_auth import install_operator_gate
        install_operator_gate(app)
    except Exception as _exc:
        import logging as _logging
        _logging.getLogger(__name__).error("operator gate install failed: %s", _exc)

    # Video surface gate: the /video arm (SPA shell + /video/* and /movie/* API
    # and media routes) sits behind the SAME auth boundary as the console — a
    # valid console session (mode-aware; permissive in `open`, enforced in
    # `external`) OR a video-scoped share credential (a stubbed seam today).
    # DELIBERATELY separate from the operator gate so the share credential can
    # NEVER authorize a console/operator route. Never break boot.
    try:
        from hugpy_server.app.video_auth import install_video_gate
        install_video_gate(app)
    except Exception as _exc:
        import logging as _logging
        _logging.getLogger(__name__).error("video gate install failed: %s", _exc)

    # Studio/Media plane gate (2026-08-06): /media, /ml, /uploads, /session and
    # /chat were gated by NOTHING — an anonymous caller could upload into the
    # shared store, run the media pipelines and spend GPU on /chat/stream. This
    # gate requires a MEMBER (an approved central account with the hugpy /
    # clownworld site grant), an operator, or a valid API key — deliberately a
    # SEPARATE gate from the console one, so a member credential can never
    # authorize a console/operator route. Never break boot.
    try:
        from hugpy_server.app.member_auth import install_member_gate
        install_member_gate(app)
    except Exception as _exc:
        import logging as _logging
        _logging.getLogger(__name__).error("member gate install failed: %s", _exc)

    # Human-friendly /endpoints: content-negotiate the abstract_flask endpoint
    # inspector so a browser hitting dev.hugpy.ai/endpoints gets a rendered,
    # searchable page while curl / programmatic clients still get the exact JSON
    # abstract_flask produced. Overrides the view in place (no new rule). Never
    # break boot.
    try:
        from hugpy_server.app.endpoints_view import install_endpoints_view
        install_endpoints_view(app)
    except Exception as _exc:
        import logging as _logging
        _logging.getLogger(__name__).error("endpoints view install failed: %s", _exc)

    # Composition: every cross-package seam in py/WIRING.md marked "server"
    # (fleet placement into the engine, oracle providers, task plugins, the
    # catalog bridge, oracle video hooks, curation providers, the storage
    # footprint selector, the HF-token listener, the control bus and the
    # eviction store sink). Each step is isolated; the report lands on
    # app.extensions["hugpy_wiring"]. Must never break boot.
    if wire:
        try:
            from hugpy_server.wiring import install_all
            install_all(app)
        except Exception as _exc:  # noqa: BLE001
            logger.error("composition wiring failed: %s", _exc)

    # Start the Video Intelligence job worker daemon ONCE per process. Each of
    # the gunicorn worker processes starts one; the bus's atomic cross-process
    # claim guarantees exactly one daemon runs any given job — that's intended.
    # Guarded by a module-level boolean (no double-start on re-entrant app
    # creation) AND wrapped in try/except so a failure here logs and can NEVER
    # break app creation or the existing surfaces.
    global _VIDEO_DAEMON_STARTED
    if not _VIDEO_DAEMON_STARTED and _daemons_enabled(start_daemons):
        try:
            from hugpy_video.intel import media_bus as _media_bus
            _media_bus.start_worker_daemon()
            _VIDEO_DAEMON_STARTED = True
        except Exception as _exc:
            import logging as _logging
            _logging.getLogger(__name__).error(
                "video_intel worker daemon start failed: %s", _exc
            )
    # POST-DOWNLOAD ADMISSION runner (2026-09-23): claims the admission jobs
    # every completed download enqueues (hugpy_storage.admission) and runs the
    # static audit -> benchmark -> admitted/held gate (hugpy_ops.admission).
    # One runner per host (flock-elected inside), so every gunicorn worker may
    # call this. hugpy-ops is an optional extra of the server: without it the
    # jobs stay queued and GET /llm/admission says so.
    global _ADMISSION_RUNNER_STARTED
    if not _ADMISSION_RUNNER_STARTED and _daemons_enabled(start_daemons):
        try:
            from hugpy_ops.admission import start_admission_runner
            _ADMISSION_RUNNER_STARTED = bool(start_admission_runner())
        except ImportError as _exc:
            logger.info("admission runner not started (hugpy-ops not installed: %s)", _exc)
        except Exception as _exc:  # noqa: BLE001 — must never break app creation
            logger.error("admission runner start failed: %s", _exc)
    # BENCHMARK RESUME (2026-09-23): a central restart (package promotion) no
    # longer ends a running capacity benchmark as "interrupted" — once this
    # server answers /health, the persisted run is continued with its remaining
    # lanes (flock-elected: one gunicorn worker resumes it).
    if _daemons_enabled(start_daemons):
        try:
            from hugpy_server.app.routes.review_routes import start_benchmark_resume
            start_benchmark_resume()
        except Exception as _exc:  # noqa: BLE001 — must never break app creation
            logger.error("benchmark resume hook failed: %s", _exc)
    return app


def create_app(name=None, allowed_origins=None, debug=False, **kw):
    """Flask-conventional factory name; same as ``get_hugpy_flask``."""
    return get_hugpy_flask(name=name, allowed_origins=allowed_origins,
                           debug=debug, **kw)


_APP = None


def __getattr__(name):
    """``hugpy_server.wsgi_app:app`` for gunicorn/waitress command lines —
    built lazily on first access, so importing this module stays cheap."""
    if name == "app":
        global _APP
        if _APP is None:
            _APP = get_hugpy_flask()
        return _APP
    raise AttributeError(name)


def _serve(flask_app, host: str, port: int, threads: int, workers: int,
           debug: bool) -> int:
    """gunicorn on POSIX, waitress on Windows, Flask dev server as the last
    resort. Server-agnostic glue; no routes live here."""
    bind = f"{host}:{port}"
    banner = (f"hugpy serving on http://{bind}  (console at /, API at /api/v1)\n"
              f"  first run? finish setup at  http://{bind}/welcome")
    if os.name == "posix":
        try:
            from gunicorn.app.base import BaseApplication
        except ImportError:
            BaseApplication = None
        if BaseApplication is not None:
            class _App(BaseApplication):
                def load_config(self):
                    self.cfg.set("bind", bind)
                    self.cfg.set("workers", max(1, int(workers)))
                    self.cfg.set("threads", max(1, int(threads)))
                    self.cfg.set("timeout", 300)

                def load(self):
                    return flask_app

            print(banner)
            _App().run()
            return 0
    try:
        from waitress import serve as _waitress_serve
    except ImportError:
        print(f"hugpy: gunicorn/waitress not installed; using the Flask dev "
              f"server on {bind}", file=sys.stderr)
        flask_app.run(host=host, port=port, debug=debug)
        return 0
    print(banner + "  [waitress]")
    _waitress_serve(flask_app, host=host, port=port, threads=threads)
    return 0


def build_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(
        prog="hugpy-serve",
        description="Run the hugpy console + API in one process.")
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7002, help="bind port (default: 7002)")
    parser.add_argument("--threads", type=int, default=8, help="server threads (default: 8)")
    parser.add_argument("--workers", type=int, default=1,
                        help="gunicorn worker processes (default: 1 — the "
                             "registries and job store are per-process singletons)")
    parser.add_argument("--auth", choices=("open", "external"),
                        help="auth mode (default: open, or HUGPY_AUTH_MODE)")
    parser.add_argument("--origins", help="comma-separated CORS origins (default: same-origin only)")
    parser.add_argument("--debug", action="store_true")
    return parser


def main(argv=None) -> int:
    """``hugpy-serve`` console script (also what ``hugpy serve`` dispatches
    to). Builds the app — which runs ``wiring.install_all()`` — then serves."""
    args = build_arg_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.auth:
        os.environ["HUGPY_AUTH_MODE"] = args.auth
    else:
        # Distribution default: single-operator instance, no login wall. The
        # /v1 API-key system still gates programmatic access.
        os.environ.setdefault("HUGPY_AUTH_MODE", "open")
    origins = [o.strip() for o in (args.origins or "").split(",") if o.strip()] or None
    flask_app = get_hugpy_flask(name="hugpy", allowed_origins=origins, debug=args.debug)
    return _serve(flask_app, args.host, args.port, args.threads, args.workers, args.debug)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
