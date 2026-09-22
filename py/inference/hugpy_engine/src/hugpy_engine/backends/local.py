"""The in-process backend: the facade's default, built from the engine's own
implementation modules (catalog, allocation, native engines, dispatch).

Every method imports its implementation lazily so ``import hugpy_engine``
stays CPU-only and free of llama_cpp / torch / transformers; the heavy stacks
load on first use of the method that needs them.
"""

from __future__ import annotations

from typing import Any, Mapping


class LocalBackend:
    """Carry a request from discovery through reply inside this process."""

    def __init__(self, *, install_bridge: bool = True) -> None:
        # Installing the catalog bridge makes hugpy_storage read the live
        # registry and lets a download refresh discovery; it is best-effort so
        # a box without storage state still answers catalog questions.
        if install_bridge:
            from hugpy_engine.catalog_bridge import install

            install()

    # -- catalog -----------------------------------------------------------
    def catalog_rows(self):
        from hugpy_engine.config.models.models_config import get_models_dict

        return get_models_dict(dict_return=True)

    def resolve_model(self, **kwargs):
        from hugpy_engine.resolvers.model_resolver import resolve_model_key

        return resolve_model_key(**kwargs)

    def refresh_models(self, *, discover=True):
        from hugpy_engine.config.models.models_config import refresh_registry

        return refresh_registry(run_discovery=discover)

    # -- allocation --------------------------------------------------------
    def feasible_allocations(self, *args, **kwargs):
        from hugpy_engine.alloc_modes import feasible_modes

        return feasible_modes(*args, **kwargs)

    def choose_allocation(self, *args, **kwargs):
        from hugpy_engine.alloc_modes import default_allocation

        return default_allocation(*args, **kwargs)

    def allocation_spill(self, mode, **options):
        from hugpy_engine.alloc_modes import mode_to_spill

        return mode_to_spill(mode, **options)

    def resource_status(self):
        from hugpy_engine.spill import describe

        return describe()

    def plan_admission(self, *args, **kwargs):
        from hugpy_engine.eviction import plan_admission

        return plan_admission(*args, **kwargs)

    # -- native engines ----------------------------------------------------
    def native_engine_status(self, *, probe=True):
        from hugpy_engine.native.resolve import native_engine_status

        return native_engine_status(probe=probe)

    def native_engine_binaries(self):
        from hugpy_engine.native.resolve import cli_bin, rpc_bin, server_bin

        return {"server": server_bin(), "rpc": rpc_bin(), "cli": cli_bin()}

    def install_native_engine(self, **options):
        from hugpy_engine.native.fetch import install

        return install(**options)

    def build_native_engine(self, **options):
        from hugpy_engine.native.build import build_from_source

        return build_from_source(**options)

    def llama_runner(self, model_key):
        from hugpy_engine.llama.runners.get import get_llama_runner

        return get_llama_runner(model_key)

    def loaded_llama_runners(self):
        from hugpy_engine.llama.runners.get import loaded_runner_detail

        return loaded_runner_detail()

    def evict_llama_runner(self, model_key):
        from hugpy_engine.llama.runners.get import evict_llama_runner

        return evict_llama_runner(model_key)

    # -- runtime -----------------------------------------------------------
    def resolve_request(self, request: Mapping[str, Any]):
        from hugpy_engine.resolvers import resolve

        return resolve(dict(request))

    def runner_for(self, model_key=None, *, task=None):
        from hugpy_engine.dispatch import runner_for

        return runner_for(model_key, task=task)

    def execute_request(self, *args, **kwargs):
        from hugpy_engine.dispatch import execute_prompt

        return execute_prompt(*args, **kwargs)

    async def stream_request(self, *args, cancel_event=None, **kwargs):
        from hugpy_engine.dispatch import execute_prompt_stream

        async for event in execute_prompt_stream(*args, cancel_event=cancel_event, **kwargs):
            yield event

    async def stream_chat_request(self, *args, cancel_event=None, **kwargs):
        from hugpy_engine.dispatch import execute_chat_stream

        async for event in execute_chat_stream(*args, cancel_event=cancel_event, **kwargs):
            yield event

    def run_sync(self, awaitable):
        from hugpy_platform import async_runtime

        return async_runtime.run(awaitable)

    def loaded_models(self):
        from hugpy_engine.dispatch import loaded_model_keys

        return tuple(loaded_model_keys())

    def loading_models(self):
        from hugpy_engine.dispatch import loading_model_keys

        return tuple(loading_model_keys())

    def evict_model(self, model_key, *, task=None):
        from hugpy_engine.dispatch import evict

        return evict(model_key, task=task)

    def clear_models(self):
        from hugpy_engine.dispatch import clear

        return clear()

    def supported_tasks(self):
        from hugpy_engine.dispatch import supported_task_keys

        return tuple(supported_task_keys())


__all__ = ["LocalBackend"]
