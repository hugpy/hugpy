"""Common constructor state for media runner adapters."""

from __future__ import annotations


class RunnerConfig:
    def __init__(self, cfg, **runtime_kwargs):
        self.cfg = cfg
        self.model_key = cfg.model_key
        self._runtime_kwargs = runtime_kwargs
