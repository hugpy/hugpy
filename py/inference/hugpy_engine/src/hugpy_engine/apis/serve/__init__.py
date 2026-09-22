"""Deprecated location — re-exports of :mod:`hugpy_engine.serve`."""
from hugpy_engine.serve.serve import (
    ServeMode,
    ServePlan,
    ServeSpec,
    apply_plan,
    build_serve_specs,
    install_serving,
    serve_endpoint,
    serve_model_name,
    serve_spec_for,
    serving_overview,
    start_serving,
    stop_serving,
)
from hugpy_engine.serve.serve_cli import main

__all__ = [
    "ServeMode", "ServePlan", "ServeSpec", "apply_plan", "build_serve_specs",
    "install_serving", "main", "serve_endpoint", "serve_model_name", "serve_spec_for",
    "serving_overview", "start_serving", "stop_serving",
]
