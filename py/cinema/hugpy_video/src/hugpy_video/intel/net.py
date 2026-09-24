"""URL normalization shared by the studio runner and reservation engine."""

from __future__ import annotations

from urllib.parse import urlparse


def url_host(url: str) -> str:
    """Return a lower-case host, accepting URLs without a scheme."""
    if not url:
        return ""
    normalized = url if "://" in url else "http://" + url
    try:
        return (urlparse(normalized).hostname or "").lower()
    except Exception:  # noqa: BLE001
        return ""
