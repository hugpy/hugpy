"""Deprecated location — re-export of the canonical serve module.

This used to be a near-byte-for-byte duplicate of ``hugpy_engine.serve.serve``
carrying its own (Linux-only, hardcoded) ``LLAMA_CPP_DIR``/systemd defaults.
Those have been unified and made cross-platform in the canonical module; this
shim re-exports it so any lingering import path keeps working while there is
exactly one implementation.
"""
from hugpy_engine.serve.serve import (  # noqa: F401
    DEFAULT_LLAMA_NGL,
    DEFAULT_LLAMA_THREADS,
    DEFAULT_SERVE_MODE,
    LLAMA_CPP_DIR,
    LLAMA_SERVER_BIN,
    OffDriver,
    ServeMode,
    ServePlan,
    ServeSpec,
    SupervisedDriver,
    SwapDriver,
    SystemdDriver,
    apply_plan,
    build_serve_specs,
    get_serve_driver,
    install_serving,
    register_serve_driver,
    serve_endpoint,
    serve_model_name,
    serve_spec_for,
    serving_overview,
    start_serving,
    stop_serving,
)
