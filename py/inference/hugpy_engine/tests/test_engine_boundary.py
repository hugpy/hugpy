from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

import hugpy_engine as engine


class FakeBackend:
    def __init__(self):
        self.seen = {}
        self.rows = {
            "vision": {
                "name": "Vision",
                "framework": "gguf",
                "tasks": ["image-text-to-text", "text-generation"],
                "primary_task": "image-text-to-text",
                "model_max_length": 8192,
            }
        }

    def catalog_rows(self):
        return self.rows

    def resolve_model(self, **kwargs):
        self.seen["resolve"] = kwargs
        return "vision"

    async def stream_chat_request(self, *args, **kwargs):
        self.seen["query"] = kwargs
        yield SimpleNamespace(type="token", text="hello ")
        yield SimpleNamespace(type="token", text="world")
        yield SimpleNamespace(
            type="done",
            finish_reason="stop",
            usage={"total_tokens": 4},
            timings={"predicted_per_second": 20.0},
        )

    def run_sync(self, awaitable):
        return asyncio.run(awaitable)

    def feasible_allocations(self, *args, **kwargs):
        return ("gpu-only", "max-gpu")

    def choose_allocation(self, *args, **kwargs):
        return {"mode": "gpu-only", "spill": {"n_gpu_layers": -1}}

    def allocation_spill(self, mode, **options):
        return {"n_gpu_layers": "off"} if mode == "ram-only" else {}


@pytest.fixture
def backend():
    fake = FakeBackend()
    engine.configure_backend(fake)
    try:
        yield fake
    finally:
        engine.configure_backend(None)


def test_package_import_does_not_load_the_monolith():
    assert sys.modules.get("abstract_hugpy_dev") is None  # never a loaded module


def test_catalog_and_text_task_selection_are_standalone(backend):
    models = engine.list_models(task="text-generation")
    assert [model.key for model in models] == ["vision"]
    assert engine.resolve_model(task="text-generation") == "vision"
    assert backend.seen["resolve"]["task"] == "text-generation"


def test_prompt_runs_through_catalog_runtime_and_reply_assembly(backend):
    result = asyncio.run(
        engine.query_result(
            "say hello", model_key="vision", request_id="request-1"
        )
    )

    assert result == engine.QueryResult(
        text="hello world",
        request_id="request-1",
        finish_reason="stop",
        usage={"total_tokens": 4},
        timings={"predicted_per_second": 20.0},
    )
    assert backend.seen["query"]["task"] == "text-generation"


def test_sync_query_and_allocation_surface(backend):
    assert engine.query_sync("hello") == "hello world"
    assert "gpu-only" in engine.feasible_allocations("gguf", 1, 2, 3)
    assert engine.choose_allocation("gguf", 1, 2, 3)["mode"] == "gpu-only"
    assert engine.allocation_spill("ram-only") == {"n_gpu_layers": "off"}
