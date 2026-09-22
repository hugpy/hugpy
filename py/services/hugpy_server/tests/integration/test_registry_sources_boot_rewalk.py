"""Slice 6 — a worker's registry must be honest about on-disk models.

The ae 2026-07-17 incident (named by slice-5's histogram): scan_skip_reasons=
{no_config: 63, not_local: 10, comfy: 1}. The ~24 on-disk models (each dir
carries weights + hugpy.json) died in `no_config` — the registry had no entry
for them.

ROOT CAUSE: the worker's registry is built ONCE at module import from the
discovery REPORT FILE (<DEFAULT_ROOT>/projects/model_discovery.json). Nothing on
the worker ever re-walks the tree, so an ABSENT or STALE report leaves the
registry STAPLES-ONLY. The on-disk dirs carry per-dir hugpy.json markers — the
source of truth discover_models reads — but nothing reads them at scan time.

FIX: call refresh_registry(run_discovery=True) at worker boot (its own docstring
says to — the worker just never did), so on-disk markers re-derive their configs
regardless of the report file's state.

DIAGNOSTIC: registry_sources {staple, discovered, central, comfy, total} in the
heartbeat — a dead source (discovered==0) is visible in one beat.

Run: venv/bin/python -m pytest tests/test_registry_sources_boot_rewalk.py -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hugpy_fleet.worker import agent as A
from hugpy_fleet.worker import imports as WI
from hugpy_storage import provision as P


# ── registry_sources: honest per-origin partition ──────────────────────────
def test_registry_sources_partitions_by_origin(monkeypatch):
    from hugpy_fleet.worker.imports import models_config as mc
    # A fabricated registry with one of each origin.
    fake_reg = {
        "StapleA": {"framework": "gguf", "folder": "gguf/o/StapleA"},   # staple
        "comfy-x": {"framework": "comfy", "filename": "x.safetensors"},  # comfy
        "DiscZ":   {"framework": "gguf", "dir": "/mnt/hot990/models/gguf/o/DiscZ",
                    "folder": "gguf/o/DiscZ"},                           # discovered
        "CentQ":   {"framework": "gguf", "folder": "gguf/o/CentQ"},      # central
    }
    monkeypatch.setattr(mc, "MODELS", {"StapleA": {}})
    monkeypatch.setattr(mc, "MODEL_REGISTRY_DICT", fake_reg)
    src = A._registry_sources()
    assert src == {"staple": 1, "discovered": 1, "central": 1, "comfy": 1, "total": 4}
    # partition invariant
    assert src["total"] == (src["staple"] + src["discovered"]
                            + src["central"] + src["comfy"])


def test_registry_sources_staples_only_is_the_ae_fingerprint(monkeypatch):
    """The exact ae failure shape: registry == staples only (discovered==0,
    central==0). This is the one-beat signal that the discovery layer is dead."""
    from hugpy_fleet.worker.imports import models_config as mc
    staples = {f"Staple{i}": {"framework": "gguf", "folder": f"gguf/o/S{i}"}
               for i in range(11)}
    monkeypatch.setattr(mc, "MODELS", {k: {} for k in staples})
    monkeypatch.setattr(mc, "MODEL_REGISTRY_DICT", staples)
    src = A._registry_sources()
    assert src["staple"] == 11
    assert src["discovered"] == 0        # THE dead source
    assert src["central"] == 0
    assert src["total"] == 11


def test_registry_sources_never_raises(monkeypatch):
    from hugpy_fleet.worker.imports import models_config as mc
    monkeypatch.setattr(mc, "MODEL_REGISTRY_DICT", None)   # degenerate
    assert A._registry_sources()["total"] == 0             # no crash


# ── the fix: refresh_registry re-derives on-disk configs from the report ────
def test_refresh_registry_recovers_discovered_models_from_a_report(monkeypatch):
    """The healing step: a staples-only registry, handed a discovery report (what
    discover_models produces from on-disk hugpy.json markers), gains the on-disk
    models. This is what the boot re-walk delivers regardless of the report FILE."""
    from hugpy_fleet.worker.imports import models_config as mc

    before = set(mc.MODEL_REGISTRY.keys())
    # A discovery report row as discover_models emits it (carries `dir`).
    report = {
        "unsloth~Qwen3-Coder-Next-GGUF": {
            "name": "Qwen3-Coder-Next-GGUF",
            "hub_id": "unsloth/Qwen3-Coder-Next-GGUF",
            "framework": "gguf",
            "dir": "/mnt/hot990/hugpy-worker/models/gguf/unsloth/Qwen3-Coder-Next-GGUF",
            "folder": "gguf/unsloth/Qwen3-Coder-Next-GGUF",
            "tasks": ["text-generation"],
            "primary_task": "text-generation",
        }
    }
    # Drive the merge path directly (no disk walk): refresh_registry with a
    # supplied report is equivalent to run_discovery having produced it.
    monkeypatch.setattr(
        "hugpy_engine.apis.get_module.discover_models",
        lambda **kw: report)
    try:
        mc.refresh_registry(run_discovery=True)
        assert "unsloth~Qwen3-Coder-Next-GGUF" in mc.MODEL_REGISTRY
        # get_model_config now resolves it — no longer a no_config skip.
        from hugpy_engine.config.main import get_model_config
        cfg = get_model_config("unsloth~Qwen3-Coder-Next-GGUF")
        assert getattr(cfg, "framework", None) == "gguf"
    finally:
        # Restore a clean registry for other tests in the session.
        mc.refresh_registry(run_discovery=False)


# ── the boot path actually calls the re-walk ────────────────────────────────
def test_boot_rewalk_is_wired_into_main(monkeypatch):
    """Guard against the fix silently reverting: the worker main() must invoke
    refresh_registry(run_discovery=True) at boot. We assert the call is present
    in the boot source rather than standing up a whole worker."""
    import inspect
    src = inspect.getsource(A.main)
    assert "refresh_registry(run_discovery=True)" in src
    assert "boot registry re-walk" in src.lower()


# ── the scan skip-reason + registry_sources tell the story together ─────────
class _State:
    def __init__(self, assigned=None):
        self.assigned_models = list(assigned or [])
        self._provisioning = []


def test_no_config_skips_when_registry_lacks_the_on_disk_model(monkeypatch):
    """Reproduce the ae mechanism at the scan: an on-disk model the registry
    doesn't know fails get_model_config -> counted no_config, never a row. This
    is exactly what the boot re-walk prevents (by getting it INTO the registry)."""
    def _boom(mk):
        raise KeyError(mk)
    monkeypatch.setattr(WI, "get_models_dict", lambda: {"onDiskButUnknown": None})
    monkeypatch.setattr(WI, "get_model_config", _boom)   # registry has no entry
    monkeypatch.setattr(A, "loaded_model_keys", lambda: [])
    monkeypatch.setattr(A, "_slot_occupants", lambda *a, **k: set())
    monkeypatch.setattr(A, "_loading_model_keys", lambda: [])
    monkeypatch.setattr(A, "_models_local", lambda s: [])
    scan = A._reap_scan(_State(assigned=[]))
    assert scan["scan_rows"] == 0
    assert scan["scan_skip_reasons"].get("no_config") == 1


# ── central passes registry_sources through verbatim ────────────────────────
def test_registry_sources_survive_storage_proposal():
    from hugpy_fleet.central.workers import storage_proposal
    GIB = 1 << 30
    out = storage_proposal({
        "storage": {"cache_used_bytes": 672 * GIB, "disk_free": 900 * GIB,
                    "models": [],
                    "registry_sources": {"staple": 11, "discovered": 0,
                                         "central": 0, "comfy": 1, "total": 12}},
        "disk": {"free_bytes": 900 * GIB},
        "limits": {"disk_cache_gib": 400},
    })
    assert out["registry_sources"] == {"staple": 11, "discovered": 0,
                                       "central": 0, "comfy": 1, "total": 12}


def test_registry_sources_degrade_for_a_pre_slice6_worker():
    from hugpy_fleet.central.workers import storage_proposal
    out = storage_proposal({
        "storage": {"cache_used_bytes": 0, "disk_free": 1 << 30, "models": []},
        "disk": {}, "limits": {},
    })
    assert out["registry_sources"] == {}
