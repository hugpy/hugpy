"""
Media URL builder.

Maps local filesystem paths to public URLs based on file extension.
No model dependencies — this is pure path logic.
"""

import os
from typing import Dict, Optional

EXT_TO_PREFIX: Dict[str, str] = {
    ".png": "images",
    ".jpg": "images",
    ".jpeg": "images",
    ".gif": "images",
    ".mp4": "videos",
    ".mp3": "audio",
    ".wav": "audio",
    ".pdf": "documents",
}


def generate_media_url(
    fs_path: str,
    domain: Optional[str] = None,
    repository_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Convert a local path inside *repository_dir* into a public URL.

    Returns None when either required arg is missing or *fs_path*
    doesn't fall under *repository_dir*.
    """
    if not repository_dir or not domain:
        return None

    fs_abs = os.path.abspath(fs_path)
    repo_abs = os.path.abspath(repository_dir)

    if not fs_abs.startswith(repo_abs):
        return None

    rel = fs_abs[len(repo_abs):].lstrip(os.sep).replace(os.sep, "/")
    ext = os.path.splitext(fs_abs)[1].lower()
    prefix = EXT_TO_PREFIX.get(ext, "repository")

    return f"{domain.rstrip('/')}/{prefix}/{rel}"
