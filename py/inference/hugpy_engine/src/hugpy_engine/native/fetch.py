"""Download a prebuilt llama.cpp release for this OS/arch.

llama.cpp publishes per-platform binary zips on its GitHub releases
(``ggml-org/llama.cpp`` by default), e.g.::

    llama-b1234-bin-ubuntu-x64.zip
    llama-b1234-bin-macos-arm64.zip
    llama-b1234-bin-win-cpu-x64.zip
    llama-b1234-bin-win-cuda-12.4-x64.zip

We pick the asset matching ``platform.system()`` + ``platform.machine()`` (and
the CPU/CUDA variant), unpack it into ``paths.engine_dir()``, and mark the
binaries executable. If no asset matches (exotic arch, or a CUDA build the
release channel doesn't carry) the caller can fall back to
:func:`hugpy.engine.build.build_from_source`.

Everything is overridable by env:
    HUGPY_ENGINE_REPO   owner/repo            (default ggml-org/llama.cpp)
    HUGPY_ENGINE_TAG    release tag | latest  (default latest)
    GITHUB_TOKEN        raises the API rate limit (optional)
"""
from __future__ import annotations

import io
import os
import platform
import re
import stat
import zipfile
from typing import Optional

import requests

from hugpy_platform.platform_facade import IS_LINUX, IS_MACOS, IS_WINDOWS, env_value
from hugpy_platform.app_dirs import engine_dir
from hugpy_engine.native import resolve

_DEFAULT_REPO = "ggml-org/llama.cpp"
_API = "https://api.github.com/repos/{repo}/releases/{ref}"


def _arch_tokens() -> list[str]:
    m = (platform.machine() or "").lower()
    if m in ("x86_64", "amd64", "x64"):
        return ["x64", "x86_64", "amd64"]
    if m in ("arm64", "aarch64"):
        return ["arm64", "aarch64"]
    return [m] if m else []


def _os_token() -> str:
    if IS_WINDOWS:
        return "win"
    if IS_MACOS:
        return "macos"
    if IS_LINUX:
        return "ubuntu"   # llama.cpp labels its Linux build "ubuntu"
    return ""


def _score_asset(name: str, want_cuda: bool) -> int:
    """Higher is better; -1 means unusable for this platform."""
    n = name.lower()
    if not n.endswith(".zip") or "bin" not in n:
        return -1
    if _os_token() not in n:
        return -1
    archs = _arch_tokens()
    if archs and not any(a in n for a in archs):
        return -1
    has_cuda = "cuda" in n
    if want_cuda and not has_cuda:
        return -1
    if not want_cuda and has_cuda:
        # CPU build requested — accept a CUDA asset only as a last resort.
        return 1
    score = 10
    # Prefer plain CPU/vulkan-free builds for the default case.
    if not want_cuda and not any(t in n for t in ("vulkan", "hip", "sycl", "kompute")):
        score += 5
    return score


def _pick_asset(assets: list[dict], want_cuda: bool) -> Optional[dict]:
    best, best_score = None, 0
    for a in assets:
        s = _score_asset(a.get("name", ""), want_cuda)
        if s > best_score:
            best, best_score = a, s
    return best


def _release(repo: str, tag: str) -> dict:
    ref = "latest" if tag in ("", "latest") else f"tags/{tag}"
    headers = {"Accept": "application/vnd.github+json"}
    token = env_value("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(_API.format(repo=repo, ref=ref), headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _mark_executable(root: str) -> None:
    if IS_WINDOWS:
        return
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.startswith("llama-") or f in ("rpc-server",) or f.endswith(".so") or f.endswith(".dylib"):
                p = os.path.join(dirpath, f)
                try:
                    os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                except OSError:
                    pass


def install(*, cuda: bool = False, tag: Optional[str] = None,
            repo: Optional[str] = None, force: bool = False) -> dict:
    """Fetch + unpack the engine. Returns resolved binary paths.

    Idempotent: if ``llama-server`` is already resolvable and ``force`` is False,
    returns immediately without downloading.
    """
    if not force and resolve.server_bin():
        return _resolved(note="already installed")

    repo = repo or env_value("HUGPY_ENGINE_REPO") or _DEFAULT_REPO
    tag = tag or env_value("HUGPY_ENGINE_TAG") or "latest"

    rel = _release(repo, tag)
    asset = _pick_asset(rel.get("assets", []), want_cuda=cuda)
    if not asset:
        raise RuntimeError(
            f"no prebuilt llama.cpp asset for {_os_token()}/{platform.machine()} "
            f"(cuda={cuda}) in {repo}@{rel.get('tag_name', tag)}. "
            f"Try `hugpy install-engine --build-from-source`.")

    dest = engine_dir()
    url = asset["browser_download_url"]
    name = asset["name"]
    blob = requests.get(url, timeout=600)
    blob.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(blob.content)) as zf:
        zf.extractall(dest)
    _flatten_single_dir(dest)
    _mark_executable(dest)

    server = resolve.server_bin()
    if not server:
        raise RuntimeError(
            f"downloaded {name} into {dest} but no llama-server binary was found "
            f"inside it — the asset layout may have changed.")
    return _resolved(note=f"installed {name} ({rel.get('tag_name')})")


def _flatten_single_dir(dest: str) -> None:
    """Some zips wrap everything in a single top dir; lift its contents up so the
    resolver's flat/``bin/``/``build/bin/`` search finds the binaries."""
    try:
        entries = [e for e in os.listdir(dest) if not e.startswith(".")]
    except OSError:
        return
    if len(entries) != 1:
        return
    only = os.path.join(dest, entries[0])
    if not os.path.isdir(only):
        return
    if any(n.startswith("llama-") for n in os.listdir(only)) or \
       os.path.isdir(os.path.join(only, "build")) or os.path.isdir(os.path.join(only, "bin")):
        for child in os.listdir(only):
            src = os.path.join(only, child)
            dst = os.path.join(dest, child)
            if not os.path.exists(dst):
                os.replace(src, dst)


def _resolved(note: str) -> dict:
    server = resolve.server_bin()
    return {
        "note": note,
        "engine_dir": engine_dir(),
        "server_bin": server,
        "rpc_bin": resolve.rpc_bin(),
        "cli_bin": resolve.cli_bin(),
        # k94: record the install in the box's persisted config so a worker
        # service under a different user/HOME resolves it too (the install used
        # to persist nothing and the operator reinstalled forever). Runs on the
        # already-installed early return as well, so one install-engine run
        # repairs a box that installed before persistence existed. A system/PATH
        # llama-server is not ours to pin — nothing is recorded for it.
        "persisted_to": (resolve.persist_install(engine_dir(), server)
                         if server and resolve.managed_engine_root(server)
                         else None),
    }
