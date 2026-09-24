"""Path components shared by model catalog adapters."""

from __future__ import annotations

import re


def safe_path_part(value: str) -> str:
    value = value.strip().replace("\\", "/")
    value = re.sub(r"[^A-Za-z0-9._/\-]+", "_", value)
    value = re.sub(r"/+", "/", value)
    return value.strip("/")
