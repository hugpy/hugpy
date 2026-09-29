"""KNOWN-GOOD CONTRACT — representation-aware serving-endpoint fast-route.

Catalogue: docs/KNOWN-GOOD-CORE.md (area "model key resolution" / serving route).
Source under test: hugpy_fleet/central/workers.py
(serving_endpoint_for(model_key, fmt), _endpoint_representation, and the
heartbeat writer that records ``representation`` on each serving_endpoints row).

Operator 2026-09-29: the row-first serving fast-route
(remote.DelegatingRunner.run/stream -> _serving_for -> serving_endpoint_for) must
be representation-aware — a request pinned with model_format='gguf' may only
fast-route onto a gguf seat, 'transformers' only onto a transformers seat, while
'auto'/absent matches as before (the serving representation wins). Same
heartbeat-refreshed row is the invalidation clock; no discovery-time table.

Deterministic: worker_store.all() and _is_online are stubbed; no worker process.
"""
from __future__ import annotations

import importlib
import types

import pytest

W = importlib.import_module("hugpy_fleet.central.workers")

GGUF_URL = "http://w-gguf:9000/ops/external/chat"
TF_URL = "http://w-tf:9000/ops/external/chat"


def _seat(model_key, endpoint, representation):
    return {"model_key": model_key, "served_model": model_key,
            "endpoint": endpoint, "representation": representation}


def _worker(wid, seats):
    return {"id": wid, "status": "online",
            "serving_endpoints": {s["model_key"]: s for s in seats}}


@pytest.fixture
def fleet(monkeypatch):
    """Install a fake online fleet; each test supplies the worker rows."""
    state = {"workers": []}
    monkeypatch.setattr(W, "worker_store",
                        types.SimpleNamespace(all=lambda: list(state["workers"])))
    monkeypatch.setattr(W, "_is_online", lambda w: True)
    monkeypatch.setattr(W, "_now", lambda: 1000.0)
    return state


# ---------------------------------------------------------------------------
# Explicit model_format pins the representation of the serving fast-route
# ---------------------------------------------------------------------------
def test_format_routes_to_the_matching_representation_seat(fleet):
    """INVARIANT: with a gguf seat AND a transformers seat of the same model both
    live, model_format='gguf' fast-routes to the gguf endpoint and
    'transformers' to the transformers endpoint. The row's representation is the
    discriminator, not the served key (both are keyed 'model-x')."""
    fleet["workers"] = [
        _worker("wg", [_seat("model-x", GGUF_URL, "gguf")]),
        _worker("wt", [_seat("model-x", TF_URL, "transformers")]),
    ]
    assert W.serving_endpoint_for("model-x", fmt="gguf")["endpoint"] == GGUF_URL
    assert W.serving_endpoint_for("model-x", fmt="transformers")["endpoint"] == TF_URL


def test_auto_matches_the_single_serving_seat_unchanged(fleet):
    """INVARIANT: model_format absent/'auto' matches exactly as before — the one
    live seat wins regardless of its representation."""
    fleet["workers"] = [_worker("wg", [_seat("model-x", GGUF_URL, "gguf")])]
    assert W.serving_endpoint_for("model-x")["endpoint"] == GGUF_URL
    assert W.serving_endpoint_for("model-x", fmt="auto")["endpoint"] == GGUF_URL
    fleet["workers"] = [_worker("wt", [_seat("model-x", TF_URL, "transformers")])]
    assert W.serving_endpoint_for("model-x", fmt="auto")["endpoint"] == TF_URL


def test_format_pin_with_no_matching_seat_falls_through(fleet):
    """INVARIANT: model_format='transformers' with ONLY a gguf seat live returns
    None (fall through to normal placement, which the resolver routes to the
    right representation) — it never mis-routes onto the gguf seat."""
    fleet["workers"] = [_worker("wg", [_seat("model-x", GGUF_URL, "gguf")])]
    assert W.serving_endpoint_for("model-x", fmt="transformers") is None
    assert W.serving_endpoint_for("model-x", fmt="gguf")["endpoint"] == GGUF_URL


def test_format_filter_survives_a_missing_representation_field(fleet):
    """INVARIANT: a legacy row without a stored ``representation`` is classified
    on the fly by _endpoint_representation (name/service_kind), so the pin still
    filters correctly instead of crashing or matching blindly."""
    row = {"model_key": "coder-GGUF", "served_model": "coder-GGUF",
           "endpoint": GGUF_URL}  # no 'representation' key
    fleet["workers"] = [{"id": "w", "status": "online",
                         "serving_endpoints": {"coder-GGUF": row}}]
    assert W.serving_endpoint_for("coder-GGUF", fmt="gguf")["endpoint"] == GGUF_URL
    assert W.serving_endpoint_for("coder-GGUF", fmt="transformers") is None


# ---------------------------------------------------------------------------
# Representation derivation
# ---------------------------------------------------------------------------
def test_endpoint_representation_binary_classification():
    """INVARIANT: _endpoint_representation is binary and positively identifies
    BOTH representations — a GGUF/quant name token or a llama.cpp/ollama
    service_kind is gguf; anything else is transformers."""
    assert W._endpoint_representation("Qwen~Coder-Next-GGUF") == "gguf"
    assert W._endpoint_representation("m", {"served_model": "qwen-Q4_K_M"}) == "gguf"
    assert W._endpoint_representation("m", {"service_kind": "ollama"}) == "gguf"
    assert W._endpoint_representation("Qwen~Coder-Next") == "transformers"
    assert W._endpoint_representation("m", {"service_kind": "vllm"}) == "transformers"
