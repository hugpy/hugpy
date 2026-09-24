"""Pure formatting of byte counts for logs and operator messages."""

from __future__ import annotations


def ffmpeg_num(value) -> str:
    """Use a compact numeric argument for ffmpeg."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return repr(value) if isinstance(value, float) else str(value)


def human_bytes(value, *, empty: str = "0 B") -> str:
    """Format bytes in binary units, preserving each caller's empty label."""
    if not value:
        return empty
    units = ("B", "KB", "MB", "GB", "TB")
    number = float(value)
    index = 0
    while number >= 1024 and index < len(units) - 1:
        number /= 1024
        index += 1
    return f"{number:.1f} {units[index]}"
