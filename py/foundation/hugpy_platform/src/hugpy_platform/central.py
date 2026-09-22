"""Single source of truth for the hugpy *central* base URL.

hugpy is self-hosted-first: a bare ``pip install hugpy && hugpy serve`` stands a
central up on ``http://127.0.0.1:7002`` and every arm (bot, keeper, worker)
dials it. Historically — back when the project was smaller — each arm grew its
own env var for that one address: the bot read ``HUGPY_BASE_URL``, the keeper
``HUGPY_CENTRAL``/``HUGPY_URL``, the workers ``WORKER_CENTRAL_URL``. Setting one
didn't redirect the others.

``HUGPY_BASE_URL`` is now the canonical name. The rest are kept as **silent
aliases** (first non-empty wins, canonical first) so existing ``.env`` files and
systemd units keep working unchanged. New code and docs should use
``HUGPY_BASE_URL`` only.
"""
import os

DEFAULT_CENTRAL = "http://127.0.0.1:7002"

# Canonical first, then legacy aliases. First non-empty value wins.
CENTRAL_ENV_VARS = ("HUGPY_BASE_URL", "HUGPY_CENTRAL", "HUGPY_URL", "WORKER_CENTRAL_URL")


def central_base_url(default=DEFAULT_CENTRAL):
    """Resolve the central base URL from the env, honouring legacy aliases.

    Returns ``default`` (localhost central) when nothing is set. Pass
    ``default=None`` for callers — e.g. a remote worker — that must require an
    explicit central rather than fall back to loopback.
    """
    for name in CENTRAL_ENV_VARS:
        val = os.environ.get(name)
        if val:
            return val.rstrip("/")
    return default.rstrip("/") if default else default
