"""Slice 6 regression: discovery/derive attribution + dedupe rules.

The 2026-07-05 phantom rows ("sd-turbo" twice, hub comfy/sd-turbo with
framework=transformers/tasks=text-generation/ctx=32768 defaults) had two
causes: (1) the enrichment chain never read the hugpy.json marker, so stamped
dirs with no config.json enriched to NOTHING and derive minted defaults;
(2) the comfy checkpoint sweep's rows were appended AFTER the staple merge,
so discovery rows walking the sweep's own layout dirs had nothing to dedupe
against. Also covers the HF-canonical framework vocabulary ("gguf", not
"llama_cpp").

Runs like the other tests here:
    venv/bin/python tests/test_registry_attribution.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib

gm = importlib.import_module("hugpy_engine.apis.get_module")
mc = importlib.import_module("hugpy_engine.config.models.models_config")
cat = importlib.import_module("hugpy_engine.categories")
marker_mod = importlib.import_module(
    "hugpy_storage.hugpy_marker")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


def test_registry_attribution():
    """Converted from the script-style suite: every `check` is an assert."""


    # --- canonical framework vocabulary (llama_cpp retired) ---------------------
    check('derive: gguf blob -> "gguf"',
          mc._derive_framework("x-GGUF", "o/x-GGUF", {}) == "gguf")
    check("derive: plain -> transformers",
          mc._derive_framework("bert", "o/bert", {}) == "transformers")
    check("RUNNER_PAIRS speaks gguf",
          ("gguf", "text-generation") in cat.RUNNER_PAIRS
          and ("gguf", "image-text-to-text") in cat.RUNNER_PAIRS)
    check("RUNNER_PAIRS no longer speaks llama_cpp",
          not any(fw == "llama_cpp" for fw, _t in cat.RUNNER_PAIRS))
    check("staples stamped gguf",
          mc.MODELS["Qwen2.5-3B-Instruct-GGUF"]["framework"] == "gguf")
    check("gguf routes to the gguf folder family",
          mc._runtime_folder("gguf", "o/r") == "gguf")

    cfg, why = mc.derive_model_config_row(
        "some-model-GGUF", {"hub_id": "o/some-model-GGUF"})
    check("derived gguf row is serveable (runner pair exists)",
          cfg is not None and cfg["framework"] == "gguf" and cfg["serveable"])

    # --- hugpy.json marker is authoritative for discovery attribution -----------
    tmp = tempfile.mkdtemp(prefix="hugpy-attr-test-")
    stamped = os.path.join(tmp, "models", "misc", "text-to-image", "comfy", "sd-turbo")
    os.makedirs(stamped)
    marker_mod.write_hugpy_marker(
        stamped, hub_id="comfy/sd-turbo", name="comfy-sd-turbo",
        framework="comfy", tasks=["text-to-image", "image-to-image"],
        primary_task="text-to-image", filename="sd_turbo.safetensors",
        source="comfy-sweep")

    part = gm.resolve_hugpy_marker(stamped, "comfy/sd-turbo")
    check("marker resolver reads framework", part.get("framework") == "comfy")
    check("marker resolver reads tasks", part.get("tasks") == ["text-to-image", "image-to-image"])
    check("marker resolver reads declared name", part.get("name") == "comfy-sd-turbo")

    meta, sources = gm.enrich(stamped, "comfy/sd-turbo",
                              gm.build_resolver_chain(use_hub=False))
    check("enrich carries marker framework", meta.framework == "comfy")
    check("enrich carries marker tasks", meta.tasks == ["text-to-image", "image-to-image"])
    check("enrich attributes the marker source", sources.get("framework") == "hugpy_marker")

    # a marker-shaped discovery row must NEVER degrade to defaults in derive
    row = dict(meta.to_dict(), hub_id="comfy/sd-turbo", dir=stamped)
    cfg, why = mc.derive_model_config_row("comfy~sd-turbo", row)
    check("derive keeps marker framework (no transformers default)",
          cfg is not None and cfg["framework"] == "comfy")
    check("derive keeps marker tasks (no text-generation default)",
          cfg["tasks"] == ["text-to-image", "image-to-image"])

    # --- storage-path layout is the attribution floor ---------------------------
    _home_orig = gm.MODELS_HOME
    try:
        gm.MODELS_HOME = os.path.join(tmp, "models")
        unstamped = os.path.join(tmp, "models", "gguf", "text-generation", "own", "repo")
        os.makedirs(unstamped)
        part = gm.resolve_layout_path(unstamped, "own/repo")
        check("layout path derives the task", part.get("tasks") == ["text-generation"])
        check("layout path derives gguf framework", part.get("framework") == "gguf")

        part = gm.resolve_layout_path(stamped, "comfy/sd-turbo")
        check("misc family implies NO framework", "framework" not in part)
        check("misc family still yields the task segment",
              part.get("tasks") == ["text-to-image"])

        outside = os.path.join(tmp, "elsewhere", "x")
        os.makedirs(outside)
        check("outside MODELS_HOME -> no layout attribution",
              gm.resolve_layout_path(outside, "a/b") == {})
    finally:
        gm.MODELS_HOME = _home_orig

    # --- dedupe: discovery must merge into sweep/staple rows, never duplicate ---
    base = {
        "comfy-sd-turbo": {                      # the sweep row (correct identity)
            "model_max_length": 77, "include": None, "name": "comfy-sd-turbo",
            "framework": "comfy", "hub_id": "comfy/sd-turbo",
            "filename": "sd_turbo.safetensors", "folder": "comfy/sd-turbo",
            "tasks": ["text-to-image", "image-to-image"],
            "primary_task": "text-to-image", "port": None,
        },
    }
    discovery = {
        "comfy~sd-turbo": {                      # discovery walking the layout dir
            "name": "sd-turbo", "hub_id": "comfy/sd-turbo",
            "dir": stamped, "folder": "misc/text-to-image/comfy/sd-turbo",
        },
    }
    merged, dropped = mc.merge_discovery_into_models(discovery, base=base)
    check("same-hub discovery row merges into the existing row",
          "comfy~sd-turbo" not in merged)
    check("merge is recorded in the drop log",
          any(k == "comfy~sd-turbo" and "same hub_id" in why for k, why in dropped))
    check("the surviving row keeps its true attribution",
          merged["comfy-sd-turbo"]["framework"] == "comfy"
          and merged["comfy-sd-turbo"]["tasks"] == ["text-to-image", "image-to-image"])
    check("exactly one row carries the hub_id",
          sum(1 for v in merged.values()
              if v.get("hub_id") == "comfy/sd-turbo") == 1)
    check("no surviving row shows default text-generation for this hub",
          all(v.get("tasks") != ["text-generation"]
              for v in merged.values() if v.get("hub_id") == "comfy/sd-turbo"))

    print(f"\nall {ok} checks passed")
