"""Canonical hugpy credential/config loader — the single-source-of-truth seam.

Precedence (same contract as hugpy_agent.config, later layer wins):
    agent.toml  ->  .env (workspace/CWD)  ->  process env  ->  explicit overrides

Canonical variable names (frozen per ARMS-ASSESSMENT-2026-08-20):
    HUGPY_API_KEY   the hp_ public API key
    HUGPY_BASE      central base URL (default https://dev.hugpy.ai/api)

Legacy aliases are accepted, lowest priority within their layer, and recorded in
Config.deprecations so callers can warn. New code must never read the aliases.
Fleet-scope store: /srv/vm_mgr/secrets/hugpy.env (0600) — pass as env_file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BASE = "https://dev.hugpy.ai/api"

# alias -> canonical; order matters only for reporting
_KEY_ALIASES = ("CONSOLE_API", "HUGPY_API_TOKEN", "HUGPY_AGENT_API", "HUGPY_API_KEY_NEWVM")
_BASE_ALIASES = ("HUGPY_BASE_URL", "HUGPY_URL", "HUGPY_CENTRAL")


@dataclass
class Config:
    base: str = DEFAULT_BASE
    api_key: str = ""
    sources: dict = field(default_factory=dict)       # attr -> winning layer
    deprecations: list = field(default_factory=list)  # (alias, layer) hits


def _parse_env_file(path: Path) -> dict:
    out = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip().strip("'\"")
    except OSError:
        pass
    return out


def _apply(cfg: Config, mapping: dict, layer: str) -> None:
    # aliases first so a canonical name in the same layer wins
    for alias in _KEY_ALIASES:
        if mapping.get(alias):
            cfg.api_key = mapping[alias]
            cfg.sources["api_key"] = layer
            cfg.deprecations.append((alias, layer))
    for alias in _BASE_ALIASES:
        if mapping.get(alias):
            cfg.base = mapping[alias]
            cfg.sources["base"] = layer
            cfg.deprecations.append((alias, layer))
    if mapping.get("HUGPY_API_KEY"):
        cfg.api_key = mapping["HUGPY_API_KEY"]
        cfg.sources["api_key"] = layer
    if mapping.get("HUGPY_BASE"):
        cfg.base = mapping["HUGPY_BASE"]
        cfg.sources["base"] = layer


def load_config(env_file: str | os.PathLike | None = None, **overrides) -> Config:
    cfg = Config()
    if env_file:
        _apply(cfg, _parse_env_file(Path(env_file)), f".env:{env_file}")
    _apply(cfg, dict(os.environ), "env")
    for attr, value in overrides.items():
        if value:
            setattr(cfg, attr, value)
            cfg.sources[attr] = "override"
    return cfg
