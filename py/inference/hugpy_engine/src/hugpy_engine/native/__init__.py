"""Native llama.cpp engine provisioning.

Two engines back hugpy's GGUF inference:

  * **in-process** — ``llama-cpp-python`` (a base dependency); needs no native
    binary and is the portable baseline used by ``mode=off`` / the Python runner.
  * **native server** — the ``llama-server`` / ``rpc-server`` executables from the
    llama.cpp C++ project, used by the always-on serve drivers and the GPU shard
    fleet. These are *not* on PyPI; this package fetches the right prebuilt
    release for the current OS/arch (or builds from source) on demand.

Public surface:

    resolve.server_bin() / rpc_bin() / cli_bin()   — locate an executable, or None
    fetch.install(...)                              — download + unpack a release
    build.build_from_source(...)                    — cmake fallback

The CLI exposes this as ``hugpy install-engine``.
"""
