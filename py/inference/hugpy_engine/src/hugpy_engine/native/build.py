"""Build llama.cpp from source via cmake — the fallback when no prebuilt asset
matches (exotic arch, or a CUDA/RPC build the release channel doesn't carry).

Requires ``git`` and ``cmake`` on PATH (and a CUDA toolkit for ``cuda=True``).
Clones into ``engine_dir()/src`` and builds into ``engine_dir()/build`` so the
resolver's ``build/bin`` search picks up ``llama-server`` / ``rpc-server``.

``-DGGML_RPC=ON`` is always set so the cross-machine shard backend
(``rpc-server``) is produced — the worker fleet needs it. This mirrors the build
hint the worker agent already prints (``cmake -DGGML_CUDA=on -DGGML_RPC=ON``).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional

from hugpy_platform.platform_facade import env_value
from hugpy_platform.binaries import resolve_bin
from hugpy_platform.app_dirs import engine_dir
from hugpy_engine.native import resolve

_DEFAULT_GIT_URL = "https://github.com/ggml-org/llama.cpp.git"


def _require(tool: str) -> str:
    path = resolve_bin(tool)
    if not path:
        raise RuntimeError(f"`{tool}` not found on PATH — needed to build the engine from source.")
    return path


def build_from_source(*, cuda: bool = False, tag: Optional[str] = None,
                      git_url: Optional[str] = None, jobs: Optional[int] = None) -> dict:
    git = _require("git")
    cmake = _require("cmake")
    git_url = git_url or env_value("HUGPY_ENGINE_GIT_URL") or _DEFAULT_GIT_URL
    tag = tag or env_value("HUGPY_ENGINE_TAG")

    root = engine_dir()
    src = os.path.join(root, "src")
    build = os.path.join(root, "build")

    if not os.path.isdir(os.path.join(src, ".git")):
        if os.path.exists(src):
            shutil.rmtree(src, ignore_errors=True)
        subprocess.run([git, "clone", "--depth", "1",
                        *(["--branch", tag] if tag else []), git_url, src], check=True)
    else:
        subprocess.run([git, "-C", src, "fetch", "--depth", "1", "origin",
                        *( [tag] if tag else [] )], check=False)

    configure = [cmake, "-S", src, "-B", build, "-DLLAMA_BUILD_SERVER=ON", "-DGGML_RPC=ON"]
    if cuda:
        configure.append("-DGGML_CUDA=on")
    subprocess.run(configure, check=True)

    # Upstream renamed the RPC backend target ``rpc-server`` -> ``ggml-rpc-server``
    # (llama.cpp tools/rpc, 2026-08). Build llama-server first, then whichever RPC
    # target this tree knows; the resolver looks for ``rpc-server`` by name, so a
    # ``ggml-rpc-server`` binary gets a same-dir ``rpc-server`` symlink.
    # (keeper 2026-08-20: found live on a-brain — the old hardcoded pair failed
    # every source build with "No rule to make target 'rpc-server'".)
    jobs_args = ["-j", str(jobs)] if jobs else []
    subprocess.run([cmake, "--build", build, "--config", "Release",
                    "--target", "llama-server"] + jobs_args, check=True)
    rpc_err = None
    for target in ("rpc-server", "ggml-rpc-server"):
        try:
            subprocess.run([cmake, "--build", build, "--config", "Release",
                            "--target", target] + jobs_args, check=True)
        except subprocess.CalledProcessError as exc:
            rpc_err = exc
            continue
        bin_dir = os.path.join(build, "bin")
        built = os.path.join(bin_dir, target)
        alias = os.path.join(bin_dir, "rpc-server")
        if target != "rpc-server" and os.path.isfile(built) and not os.path.exists(alias):
            os.symlink(target, alias)
        rpc_err = None
        break
    if rpc_err is not None:
        raise rpc_err

    server = resolve.server_bin()
    if not server:
        raise RuntimeError(f"build completed but no llama-server found under {build}")
    return {
        "note": "built from source",
        "engine_dir": root,
        "server_bin": server,
        "rpc_bin": resolve.rpc_bin(),
        "cli_bin": resolve.cli_bin(),
        # k94: same persistence as fetch.install — see fetch._resolved.
        "persisted_to": (resolve.persist_install(root, server)
                         if resolve.managed_engine_root(server) else None),
    }
