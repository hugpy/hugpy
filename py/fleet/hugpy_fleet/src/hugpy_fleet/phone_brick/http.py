"""Small JSON POST used by phone registration and RPC."""

from __future__ import annotations

import json
import urllib.request


def post_json(url: str, body: dict, timeout: float) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return resp.status, (json.loads(raw) if raw else {})
