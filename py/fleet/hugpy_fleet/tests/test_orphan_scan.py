"""Worker-side orphan (unattributed-on-disk) scan (2026-07-17 addendum, and the
2026-07-17 over-report hardening that followed it).

computron held 5.7G of a STALLED partial download (Qwen2.5-VL-3B, all *.part
files) for a model NOT allocated to it — and it appeared NOWHERE in the UI
because the reaper survey only looks at KNOWN/assigned keys. _orphan_scan walks
the store root for exactly this residue: leftover model dirs + stalled *.part
sets that match no current model. This locks its behavior against a temp store.

The SAME-DAY follow-up: ae reported 14 items / 440GB "unattributed on disk",
but 13/14 resolved to DESIGNATED catalog models. Root cause — an on-disk dir
under the flat layout (``<runtime>/<owner>/<repo>``) was compared against
assignment/catalog keys shaped ``<owner>~<repo>`` (the ``~`` qualifier
discover_models() mints on an owner-name collision) with only a
lowercase/strip normalization: ``~`` was never translated to/from ``/``, so
the two forms could only ever match by accident (a bare-repo-name fallback).
misc/comfy/** (operator doctrine: comfy is excluded from allocations) wasn't
excluded at all. Both are fixed via provision.known_model_dir_forms /
dir_is_known_model / is_doctrine_excluded — the SAME helper model_is_local's
callers are meant to converge on, so the two locality heads can't diverge.

Runs under pytest: venv/bin/python -m pytest tests/test_orphan_scan.py
"""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Import the agent module; the flat store layout is family/owner/repo under root.
from hugpy_fleet.worker import agent as A
from hugpy_fleet.worker import imports as WI


class _State:
    assigned_models = []


def _write(path, size=1024):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * size)


def _make_store():
    root = tempfile.mkdtemp()
    tr = os.path.join(root, "transformers")
    # ATTRIBUTED + resident: a real model dir for an assigned model.
    _write(os.path.join(tr, "google", "flan-t5-xl", "model.safetensors"), 4000)
    _write(os.path.join(tr, "google", "flan-t5-xl", "config.json"), 100)
    # ORPHAN 1: stalled partial — only .part files, no real weights.
    _write(os.path.join(tr, "Qwen", "Qwen2.5-VL-3B", "model.safetensors.part"), 5700)
    _write(os.path.join(tr, "Qwen", "Qwen2.5-VL-3B", "config.json.part"), 50)
    # ORPHAN 2: a completed but unassigned leftover model dir.
    _write(os.path.join(tr, "old", "leftover", "model.safetensors"), 900)
    _write(os.path.join(tr, "old", "leftover", "config.json"), 100)
    return root


def _run(root, known_hub_ids):
    # Point the scan's root resolver at our temp store.
    orig = A._models_store_root
    A._models_store_root = lambda: root
    # storage's known_model_dir_forms expands candidate dirs against the
    # platform MODELS_HOME; point it at the temp store too.
    import hugpy_platform.constants as _pc
    import hugpy_storage.model_paths as _mp
    orig_home = _pc.MODELS_HOME
    _pc.MODELS_HOME = root
    # is_model_dir's owner-level rule measures depth against get_model_home().
    orig_get_home = _mp.get_model_home
    _mp.get_model_home = lambda *a, **kw: root
    A._ORPHAN_CACHE["value"] = None      # bypass the TTL cache between runs
    try:
        return A._orphan_scan(_State(), set(known_hub_ids))
    finally:
        A._models_store_root = orig
        _pc.MODELS_HOME = orig_home
        _mp.get_model_home = orig_get_home
        A._ORPHAN_CACHE["value"] = None


def test_stalled_part_and_leftover_are_orphaned():
    root = _make_store()
    # flan-t5-xl IS known/assigned; the other two are not.
    out = _run(root, {"google/flan-t5-xl", "flan-t5-xl"})
    paths = {i["path"]: i for i in out["items"]}
    # the assigned model is NOT flagged
    assert not any("flan-t5-xl" in p for p in paths)
    # both orphans ARE flagged, with the right kind
    qwen = next((v for p, v in paths.items() if "Qwen2.5-VL-3B" in p), None)
    leftover = next((v for p, v in paths.items() if "leftover" in p), None)
    assert qwen is not None and qwen["kind"] == "partial"
    assert leftover is not None and leftover["kind"] == "stale-dir"
    assert out["count"] == 2
    assert out["bytes"] == qwen["bytes"] + leftover["bytes"]


def test_all_known_no_orphans():
    root = _make_store()
    out = _run(root, {"google/flan-t5-xl", "flan-t5-xl",
                      "Qwen/Qwen2.5-VL-3B", "Qwen2.5-VL-3B",
                      "old/leftover", "leftover"})
    assert out["count"] == 0 and out["bytes"] == 0


def test_no_store_root_is_safe():
    orig = A._models_store_root
    A._models_store_root = lambda: None
    A._ORPHAN_CACHE["value"] = None
    try:
        out = A._orphan_scan(_State(), set())
        assert out == {"items": [], "bytes": 0, "count": 0}
    finally:
        A._models_store_root = orig
        A._ORPHAN_CACHE["value"] = None


# ---------------------------------------------------------------------------
# 2026-07-17 over-report hardening: ~-key vs flat-path normalization, the
# misc/comfy policy exclusion, a loaded-but-not-designated model, and a legacy
# nested-path copy of a known model.
# ---------------------------------------------------------------------------

_ORIG_CATALOG_GET = None


def _restore_catalog_get():
    import hugpy_storage.provision as _prov
    if _ORIG_CATALOG_GET is not None:
        _prov.catalog_get = _ORIG_CATALOG_GET


def _patch_get_model_config(cfg_by_key: dict):
    """Monkeypatch worker_agent.imports.get_model_config (what
    provision.known_model_dir_forms looks up via a fresh `from .imports
    import get_model_config` each call) so tests don't need a real registry
    entry. Returns the original for restoration."""
    orig = WI.get_model_config

    def _fake(key, **kw):
        if key in cfg_by_key:
            return cfg_by_key[key]
        raise KeyError(key)

    WI.get_model_config = _fake
    import hugpy_storage.provision as _prov
    global _ORIG_CATALOG_GET
    if _ORIG_CATALOG_GET is None:
        _ORIG_CATALOG_GET = _prov.catalog_get
    _prov.catalog_get = lambda key: cfg_by_key.get(key)
    return orig


def test_tilde_qualified_key_at_flat_path_not_orphaned():
    """A designated ``<owner>~<repo>`` assignment/catalog key sitting at its
    real flat on-disk path ``<runtime>/<owner>/<repo>`` must NOT be reported —
    this is the exact ae shape (Qwen~Qwen3-Coder-Next-GGUF at
    gguf/Qwen/Qwen3-Coder-Next-GGUF). No hub_id/bare form is supplied in
    known_keys — ONLY the ~-qualified key — so this pins the ~ vs / expansion
    itself, not the bare-name fallback that used to paper over it."""
    root = tempfile.mkdtemp()
    _write(os.path.join(root, "gguf", "Qwen", "Qwen3-Coder-Next-GGUF", "model.gguf"), 2000)
    # stamp a real GGUF header so is_model_dir / any header check would pass
    p = os.path.join(root, "gguf", "Qwen", "Qwen3-Coder-Next-GGUF", "model.gguf")
    with open(p, "r+b") as f:
        f.write(b"GGUF")
    out = _run(root, {"Qwen~Qwen3-Coder-Next-GGUF"})
    assert out["count"] == 0 and out["items"] == []


def test_loaded_model_not_orphaned():
    """A currently-LOADED model (its key folded into known_keys the same way
    _worker_storage's caller does — loaded set unioned in) at its flat path is
    not orphaned, mirroring the ae case where the 14th non-comfy item was
    actually the live-loaded Qwen3-Coder-Next-GGUF."""
    root = tempfile.mkdtemp()
    _write(os.path.join(root, "gguf", "Qwen", "Qwen3-Coder-Next-GGUF", "model.gguf"), 2000)
    loaded = {"Qwen~Qwen3-Coder-Next-GGUF"}
    out = _run(root, loaded)
    assert out["count"] == 0


def test_misc_comfy_never_orphaned():
    """misc/comfy/** is excluded BY POLICY regardless of catalog membership —
    operator doctrine: comfy is excluded from allocations, models can sit on
    the drive unattributed. known_keys is empty on purpose: comfy must be
    excluded even with zero catalog knowledge of it."""
    root = tempfile.mkdtemp()
    _write(os.path.join(root, "misc", "comfy", "ae", "checkpoint.safetensors"), 50_000)
    out = _run(root, set())
    assert out["count"] == 0 and out["items"] == []


def test_unknown_dir_still_orphaned():
    """Honesty check: the narrowing must not blind the scan. A dir with no
    plausible relation to any known key is still reported, comfy-exclusion
    and ~-expansion notwithstanding."""
    root = tempfile.mkdtemp()
    _write(os.path.join(root, "gguf", "NoOwner", "TotallyUnknownModel", "model.gguf"), 3000)
    out = _run(root, {"Qwen~Qwen3-Coder-Next-GGUF", "google/flan-t5-xl"})
    assert out["count"] == 1
    assert "TotallyUnknownModel" in out["items"][0]["path"]


def test_stalled_part_of_unknown_model_still_orphaned():
    """A stalled *.part set for a model NOT in known_keys is still reported —
    the original 2026-07-17 feature this scan exists for must survive the
    narrowing."""
    root = tempfile.mkdtemp()
    _write(os.path.join(root, "gguf", "Nobody", "AbandonedPull",
                        "model.gguf.part"), 7000)
    out = _run(root, {"Qwen~Qwen3-Coder-Next-GGUF"})
    assert out["count"] == 1
    assert out["items"][0]["kind"] == "partial"
    assert "AbandonedPull" in out["items"][0]["path"]


def test_legacy_nested_path_copy_of_known_model_not_orphaned():
    """A known model whose files sit under a LEGACY task-dir shape
    (``<runtime>/<task>/<owner>/<repo>``, pre-flat-migration) must read as
    legacy-path, never orphan — candidate_model_dirs already enumerates that
    shape for a resolvable key; known_model_dir_forms folds every one of those
    relative paths in. Needs a registry entry so the resolver can compute the
    candidate set for this model's framework/hub_id."""
    root = tempfile.mkdtemp()
    legacy_dir = os.path.join(root, "gguf", "text-generation", "Qwen",
                              "Qwen3-Coder-Next-GGUF")
    _write(os.path.join(legacy_dir, "model.gguf"), 2000)
    p = os.path.join(legacy_dir, "model.gguf")
    with open(p, "r+b") as f:
        f.write(b"GGUF")

    cfg = SimpleNamespace(
        hub_id="Qwen/Qwen3-Coder-Next-GGUF", framework="gguf", filename=None,
        include=None, primary_task="text-generation",
        tasks=["text-generation"], folder="gguf/Qwen/Qwen3-Coder-Next-GGUF",
        dir=None,
    )
    orig = _patch_get_model_config({"Qwen~Qwen3-Coder-Next-GGUF": cfg})
    try:
        out = _run(root, {"Qwen~Qwen3-Coder-Next-GGUF"})
    finally:
        WI.get_model_config = orig
        _restore_catalog_get()
    assert out["count"] == 0, out["items"]
