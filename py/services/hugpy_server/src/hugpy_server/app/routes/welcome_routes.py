"""Self-hoster readiness probe — the data behind the React Landing onboarding.

hugpy is self-hosted-first (``pip install hugpy && hugpy serve`` → a central on
``127.0.0.1:7002``). The *page* a fresh operator sees is the React ``<Landing>``
(ui/src/components/Landing, routed at /welcome in the SPA, bundled into the wheel
as console_dist) — NOT a Flask page. This module only exposes the live state that
Landing renders, so it can guide ("storage empty → pull a model", "no slot serving
→ start one") instead of letting the UI walk into errors:

  GET /readiness   JSON snapshot of what's set up vs missing. NEVER 5xxes — every
                   probe is caught and reported as state. Top-level path, not
                   /api/* (the /api space is dual-mounted + SPA-catch-all'd, which
                   shadows a static /api/readiness). When the React Landing wires
                   this in, fetch it from the same origin.

NOTE: this module intentionally does NOT serve an HTML /welcome — the React
Landing owns that route. A Flask /welcome would shadow the bundled SPA's Landing
in the single-process local product, so it was removed.

Surfaces the three "pointables": the central URL (HUGPY_BASE_URL), the storage
root (DEFAULT_ROOT / settings.storage_root), and an optional *delegated* console
endpoint (HUGPY_CONSOLE_URL) — a SEPARATE @hugpy/console broker on its own port;
the UI links/hands off to it, never embeds or launches a PTY.
"""
import os
import shutil

from abstract_flask import get_bp
from flask import jsonify, request
from hugpy_engine.config.models.models_config import get_models_dict
from hugpy_engine.wire.config_schemas import settings

welcome_bp, logger = get_bp("welcome_bp", __name__)


# Coarse HTTP API contract. Bump ONLY on a breaking change to the routes/shapes a
# client depends on — decoupled from the patch-level `version`, so a consumer pins
# `api` for compatibility and reads `version` for display/telemetry.
_API_CONTRACT = 1


def _central_version():
    # Prefer the running-source __version__ (authoritative even when the installed
    # dist metadata is stale, e.g. a run-from-source dev box); fall back to metadata.
    try:
        from hugpy_server import __version__ as _v
        if _v:
            return _v
    except Exception:
        pass
    try:
        from importlib.metadata import version
        return version("hugpy-server")
    except Exception:
        return None


def _storage_state():
    try:
        root = str(getattr(settings, "storage_root", "") or os.environ.get("DEFAULT_ROOT", ""))
        exists = bool(root) and os.path.isdir(root)
        free_gb = None
        if exists:
            free_gb = round(shutil.disk_usage(root).free / (1024 ** 3), 1)
        writable = exists and os.access(root, os.W_OK)
        try:
            model_count = len(get_models_dict(dict_return=True) or {})
        except Exception:
            model_count = None
        return {"root": root or None, "exists": exists, "writable": writable,
                "free_gb": free_gb, "model_count": model_count}
    except Exception as exc:  # never let the probe break the page
        return {"error": str(exc)}


def _serving_state():
    try:
        from hugpy_engine.serve.slots import SlotPool, slots_enabled
        if not slots_enabled():
            return {"enabled": False, "any_serving": False, "slots": []}
        slots = SlotPool().overview() or []
        any_serving = any(s.get("healthy") and s.get("model_key") for s in slots)
        return {"enabled": True, "any_serving": any_serving, "slots": slots}
    except Exception as exc:
        return {"enabled": None, "any_serving": False, "error": str(exc)}


def _console_state():
    url = (os.environ.get("HUGPY_CONSOLE_URL") or "").strip().rstrip("/")
    return {"configured": bool(url), "url": url or None}


_NO_STORE = {"Cache-Control": "no-store, max-age=0"}


@welcome_bp.route("/readiness", methods=["GET"])
def readiness():
    """A snapshot the landing renders. Wrapped so it cannot 5xx."""
    try:
        base_url = request.host_url.rstrip("/")
    except Exception:
        base_url = None
    payload = {
        "central": {"version": _central_version(),
                    "auth_mode": os.environ.get("HUGPY_AUTH_MODE", "open")},
        "storage": _storage_state(),
        "serving": _serving_state(),
        "console": _console_state(),
        "connect": {"base_url": base_url},
    }
    # Live snapshot — never cache (else a stale shell hides real state).
    return jsonify(payload), 200, _NO_STORE


@welcome_bp.route("/version", methods=["GET"])
def version_info():
    """Stable, ungated version probe a client can pin to / display.

    ``api`` is the coarse contract integer (bumps only on breaking HTTP changes);
    ``version`` is the package version. A consumer asserts ``api`` for
    compatibility and shows ``version`` for telemetry — letting a downstream app
    bind to "this current hugpy version" without hardcoding model lists or routes.
    """
    return jsonify({
        "name": "hugpy-server",
        "version": _central_version(),
        "api": _API_CONTRACT,
        "auth_mode": os.environ.get("HUGPY_AUTH_MODE", "open"),
    }), 200, _NO_STORE
