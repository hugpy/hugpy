"""Bare PEFT/LoRA adapter dirs: serve them on their base, or refuse honestly.

THE BENCH FAILURE THIS GUARDS (2026-07-29, models_all_tasks.json)
    Three registry rows — qwen3.5-test-stage1-lora, Qwen2.5-1.5B-LFGRPO-300S and
    veeraragavan410~Llama-3.2-3B-sentiment — answered EVERY serve attempt with

        ValueError: Unrecognized model in <dir>.
                    Should have a `model_type` key in its config.json.

    They are bare LoRA adapter dirs: adapter_config.json + adapter_model.safetensors
    and no config.json. An adapter has no model_type because it is not a
    standalone model — it patches the base named in base_model_name_or_path.

    Two defects: discovery read only config.json (so base_model stayed None and
    the registry's adapter gate never fired), and DeepCoderConfig had no
    adapter_dir field even though DeepCoder._load_model already read one — dead
    adapter support behind a config shape that could not express it.

WHAT IS ASSERTED
    adapter + base on disk  -> load resolves to (base_dir, adapter_dir)
    adapter + base absent   -> refusal NAMING the base and the fix, not a crash
    adapter, no base named  -> refusal saying the declaration is missing
    merged checkpoint       -> NOT treated as an adapter (config.json wins)
    ordinary model dir      -> path unchanged (regression guard)
    neither config shape    -> honest "not a transformers model" refusal
    registry row            -> KEPT + serveable:False + reason, never dropped

    The heavy load is never run: what is tested is the RESOLUTION DECISION.

Runs like the other tests here:
    venv/bin/python tests/test_peft_adapter_load.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib

pa = importlib.import_module("hugpy_engine.peft_adapters")

ok = 0


def check(name, cond):
    global ok
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        raise SystemExit(f"CHECK FAILED: {name}")
    ok += 1


def test_peft_adapter_load():
    """Converted from the script-style suite: every `check` is an assert."""


    # ---------------------------------------------------------------------------
    # Fixture builders
    # ---------------------------------------------------------------------------
    def write_json(path, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)


    def make_adapter(root, rel, base_model="Qwen/Qwen3.5-0.8B", peft_type="LORA",
                     base_cls="Qwen3ForCausalLM"):
        """A bare LoRA dir, shaped like the real ones on the fleet: adapter_config
        + adapter weights + its own tokenizer, and NO config.json."""
        d = os.path.join(root, rel)
        cfg = {"peft_type": peft_type, "task_type": "CAUSAL_LM", "r": 64}
        if base_model is not None:
            cfg["base_model_name_or_path"] = base_model
        if base_cls:
            cfg["auto_mapping"] = {"base_model_class": base_cls,
                                   "unsloth_fixed": True}
        write_json(os.path.join(d, "adapter_config.json"), cfg)
        Path(os.path.join(d, "adapter_model.safetensors")).write_bytes(b"\0" * 2048)
        write_json(os.path.join(d, "tokenizer_config.json"), {"model_max_length": 32768})
        return d


    def make_model(root, rel, model_type="qwen3", weights=True):
        """An ordinary transformers model dir: config.json names a model_type."""
        d = os.path.join(root, rel)
        write_json(os.path.join(d, "config.json"),
                   {"model_type": model_type, "architectures": ["Qwen3ForCausalLM"]})
        if weights:
            Path(os.path.join(d, "model.safetensors")).write_bytes(b"\0" * 4096)
        return d


    tmp = tempfile.mkdtemp(prefix="peft-adapter-test-")

    # route_destination is peft_adapters' ONLY way to a base model's path, and it is
    # imported by name into that module — patch it to a flat map over our tmp store.
    # This keeps the test off the real store and proves resolution goes through the
    # single path chokepoint rather than guessing a layout.
    _base_dirs = {}


    def fake_route_destination(model, root=None):
        return _base_dirs.get(model.get("hub_id"),
                              os.path.join(tmp, "missing", str(model.get("hub_id"))))


    pa.route_destination = fake_route_destination


    # ---------------------------------------------------------------------------
    # 1. Adapter + base PRESENT -> serves: model_dir becomes the BASE, adapter rides
    # ---------------------------------------------------------------------------
    base_dir = make_model(tmp, "store/Qwen/Qwen3.5-0.8B")
    _base_dirs["Qwen/Qwen3.5-0.8B"] = base_dir
    adapter_dir = make_adapter(tmp, "store/armand0e/qwen3.5-test-stage1-lora")

    check("bare adapter dir is detected as an adapter", pa.is_adapter_dir(adapter_dir))
    check("adapter has no standalone config", not pa.has_standalone_config(adapter_dir))
    check("adapter base id is read from adapter_config.json",
          pa.adapter_base_id(adapter_dir) == "Qwen/Qwen3.5-0.8B")
    check("adapter peft_type is read", pa.adapter_peft_type(adapter_dir) == "LORA")
    check("present base resolves to its dir",
          pa.find_base_model_dir("Qwen/Qwen3.5-0.8B") == base_dir)
    check("base_model_present true for a base on disk",
          pa.base_model_present("Qwen/Qwen3.5-0.8B"))
    check("adapter with base on disk has NO unserveable reason",
          pa.adapter_unserveable_reason(adapter_dir) is None)
    check("resolve_adapter_pair returns (base_dir, adapter_dir)",
          pa.resolve_adapter_pair(adapter_dir) == (base_dir, adapter_dir))

    # The metadata the registry needs in order to know this is an adapter at all.
    meta = pa.adapter_metadata_fields(adapter_dir)
    check("adapter_metadata_fields reports base_model",
          meta.get("base_model") == "Qwen/Qwen3.5-0.8B")
    check("adapter_metadata_fields reports peft_type", meta.get("peft_type") == "LORA")
    check("adapter_metadata_fields carries the base architecture class",
          meta.get("architectures") == ["Qwen3ForCausalLM"])


    # ---------------------------------------------------------------------------
    # 2. Adapter + base ABSENT -> honest refusal that NAMES the base and the fix
    # ---------------------------------------------------------------------------
    orphan = make_adapter(tmp, "store/veeraragavan410/Llama-3.2-3B-sentiment",
                          base_model="unsloth/Llama-3.2-3B-Instruct",
                          base_cls="LlamaForCausalLM")
    check("absent base does not resolve",
          pa.find_base_model_dir("unsloth/Llama-3.2-3B-Instruct") is None)

    reason = pa.adapter_unserveable_reason(orphan)
    check("base-absent adapter yields a reason", bool(reason))
    check("reason NAMES the missing base model",
          "unsloth/Llama-3.2-3B-Instruct" in reason)
    check("reason states the CAUSE (it is an adapter)", "adapter" in reason.lower())
    check("reason states a FIX", "FIX:" in reason)
    check("reason is NOT the old opaque crash text",
          "Unrecognized model" not in reason and "model_type" not in reason)

    raised = None
    try:
        pa.resolve_adapter_pair(orphan)
    except pa.AdapterBaseUnavailable as exc:
        raised = exc
    check("resolve_adapter_pair raises AdapterBaseUnavailable, not ValueError",
          isinstance(raised, pa.AdapterBaseUnavailable))
    check("the exception carries the base id for re-wording",
          raised.base_model == "unsloth/Llama-3.2-3B-Instruct")
    check("the exception carries the adapter dir", raised.adapter_dir == orphan)


    # ---------------------------------------------------------------------------
    # 3. Adapter that declares NO base -> refusal says the declaration is missing
    # ---------------------------------------------------------------------------
    baseless = make_adapter(tmp, "store/nobody/baseless-lora", base_model=None,
                            base_cls=None)
    r = pa.adapter_unserveable_reason(baseless)
    check("baseless adapter yields a reason", bool(r))
    check("baseless reason names base_model_name_or_path",
          "base_model_name_or_path" in r)
    check("baseless reason states a FIX", "FIX:" in r)


    # ---------------------------------------------------------------------------
    # 4. MERGED checkpoint shipping BOTH files -> config.json wins, NOT an adapter
    #    (some trainers leave adapter_config.json beside fully-merged weights; doing
    #    the base/adapter dance there would load the wrong thing.)
    # ---------------------------------------------------------------------------
    merged = make_model(tmp, "store/someone/merged-model")
    write_json(os.path.join(merged, "adapter_config.json"),
               {"peft_type": "LORA", "base_model_name_or_path": "who/knows"})
    check("merged checkpoint is NOT an adapter", not pa.is_adapter_dir(merged))
    check("merged checkpoint resolves to itself with no adapter",
          pa.resolve_adapter_pair(merged) == (merged, None))
    check("merged checkpoint contributes no adapter metadata",
          pa.adapter_metadata_fields(merged) == {})


    # ---------------------------------------------------------------------------
    # 5. REGRESSION GUARD: an ordinary model dir is untouched
    # ---------------------------------------------------------------------------
    plain = make_model(tmp, "store/Qwen/Qwen3-Plain")
    check("ordinary dir is not an adapter", not pa.is_adapter_dir(plain))
    check("ordinary dir resolves to itself, adapter None",
          pa.resolve_adapter_pair(plain) == (plain, None))
    check("ordinary dir has no unserveable reason",
          pa.adapter_unserveable_reason(plain) is None)
    check("ordinary dir has no standalone-load refusal",
          pa.standalone_load_refusal(plain) is None)
    check("base_model_present is vacuously true when nothing is required",
          pa.base_model_present(None) and pa.base_model_present(""))


    # ---------------------------------------------------------------------------
    # 6. The FOURTH bench row's shape: neither config.json NOR adapter_config.json.
    #    Viral2AI~chatterbox is a bespoke TTS repo (loose t3_*/s3gen checkpoints)
    #    mis-registered as transformers/text-generation. It is not an adapter, so it
    #    must get its OWN honest refusal rather than the same opaque crash.
    # ---------------------------------------------------------------------------
    bespoke = os.path.join(tmp, "store/Viral2AI/chatterbox")
    os.makedirs(bespoke, exist_ok=True)
    Path(os.path.join(bespoke, "t3_cfg.safetensors")).write_bytes(b"\0" * 2048)
    Path(os.path.join(bespoke, "s3gen.pt")).write_bytes(b"\0" * 2048)

    check("bespoke repo is not an adapter", not pa.is_adapter_dir(bespoke))
    r = pa.standalone_load_refusal(bespoke)
    check("bespoke repo gets a standalone-load refusal", bool(r))
    check("refusal says what is missing (no config.json)", "config.json" in r)
    check("refusal states a FIX", "FIX:" in r)
    check("refusal is not the opaque transformers crash", "Unrecognized model" not in r)

    # A config.json that exists but names no model_type is the same class of problem.
    halfway = os.path.join(tmp, "store/half/way")
    write_json(os.path.join(halfway, "config.json"), {"architectures": ["Whatever"]})
    r = pa.standalone_load_refusal(halfway)
    check("config.json without model_type also refuses honestly", bool(r))
    check("that refusal names model_type as the missing key", "model_type" in r)


    # ---------------------------------------------------------------------------
    # 7. THE LOAD PATH: build_deepcoder_runtime must carry the decision.
    #    The heavy load is never performed — torch and the store are faked and only
    #    the resulting DeepCoderConfig is inspected.
    # ---------------------------------------------------------------------------
    cfgmod = importlib.import_module("hugpy_engine.generate.config")

    check("DeepCoderConfig can express an adapter",
          "adapter_dir" in cfgmod.DeepCoderConfig.__dataclass_fields__)


    class _FakeCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def is_bf16_supported():
            return False


    class _FakeCpu:
        @staticmethod
        def is_bf16_supported():
            return False


    class _FakeTorch:
        cuda = _FakeCuda
        cpu = _FakeCpu
        float32 = "float32"
        bfloat16 = "bfloat16"
        float16 = "float16"


    _dirs = {}
    cfgmod.require = lambda *a, **k: _FakeTorch
    cfgmod.get_model_config = lambda key: {"model_key": key}
    cfgmod.ensure_serving_weights = lambda key: _dirs[key]
    # config.py imported resolve_adapter_pair / standalone_load_refusal by name, and
    # both reach the base through peft_adapters.route_destination — already patched
    # above, so the REAL resolution logic runs against the tmp store.

    _dirs["stage1-lora"] = adapter_dir
    built = cfgmod.build_deepcoder_runtime(model_key="stage1-lora")
    check("load path rewrites model_dir to the BASE model",
          built.model_dir == base_dir)
    check("load path carries the adapter dir", built.adapter_dir == adapter_dir)
    check("adapter is part of the runtime cache key (own instance slot)",
          adapter_dir in built.cache_key())

    _dirs["plain"] = plain
    built_plain = cfgmod.build_deepcoder_runtime(model_key="plain")
    check("ordinary model still loads from its own dir", built_plain.model_dir == plain)
    check("ordinary model has no adapter_dir", built_plain.adapter_dir is None)
    check("ordinary and adapter runtimes get DIFFERENT cache keys",
          built.cache_key() != built_plain.cache_key())

    _dirs["orphan-lora"] = orphan
    err = None
    try:
        cfgmod.build_deepcoder_runtime(model_key="orphan-lora")
    except Exception as exc:  # noqa: BLE001 — asserting on the message is the point
        err = exc
    check("base-absent adapter refuses at build time", err is not None)
    check("refusal is a RuntimeError, not a transformers ValueError",
          isinstance(err, RuntimeError) and not isinstance(err, ValueError))
    check("refusal names the model key", "orphan-lora" in str(err))
    check("refusal names the base model to acquire",
          "unsloth/Llama-3.2-3B-Instruct" in str(err))
    check("refusal is not the old opaque crash",
          "Unrecognized model" not in str(err))

    _dirs["chatterbox"] = bespoke
    err = None
    try:
        cfgmod.build_deepcoder_runtime(model_key="chatterbox")
    except Exception as exc:  # noqa: BLE001
        err = exc
    check("un-loadable non-adapter dir refuses at build time too", err is not None)
    check("that refusal explains the dir isn't a transformers model",
          "config.json" in str(err) and "FIX:" in str(err))


    # ---------------------------------------------------------------------------
    # 8. THE REGISTRY: a base-absent adapter row is KEPT and flagged, never dropped.
    #    Dropping it is the silent-unavailability defect: the weights ARE on disk,
    #    and vanishing the row leaves the operator with a directory nothing explains.
    # ---------------------------------------------------------------------------
    mc = importlib.import_module(
        "hugpy_engine.config.models.models_config")

    _present = {"Qwen/Qwen3.5-0.8B"}
    mc.base_present = lambda b, *a, **k: (not b) or b in _present

    row_in = {
        "name": "Llama-3.2-3B-sentiment", "hub_id": "veeraragavan410/Llama-3.2-3B-sentiment",
        "framework": "transformers", "tasks": ["text-generation"],
        "primary_task": "text-generation", "dir": orphan,
        "base_model": "unsloth/Llama-3.2-3B-Instruct",
    }
    derived, why = mc.derive_model_config_row("veeraragavan410~Llama-3.2-3B-sentiment",
                                              dict(row_in))
    check("base-absent adapter row is NOT dropped from the registry",
          derived is not None and why is None)
    check("the kept row is flagged unserveable", derived.get("serveable") is False)
    check("the kept row names the base in its reason",
          "unsloth/Llama-3.2-3B-Instruct" in (derived.get("unserveable_reason") or ""))
    check("the kept row's reason states a FIX",
          "FIX:" in (derived.get("unserveable_reason") or ""))
    check("the kept row lists its tasks as unserveable",
          derived.get("unserveable_tasks") == ["text-generation"])
    check("the kept row still carries base_model for the load path",
          derived.get("base_model") == "unsloth/Llama-3.2-3B-Instruct")

    # ... and the same row becomes serveable the moment the base lands.
    row_in2 = dict(row_in, base_model="Qwen/Qwen3.5-0.8B",
                   hub_id="armand0e/qwen3.5-test-stage1-lora", dir=adapter_dir)
    derived2, why2 = mc.derive_model_config_row("qwen3.5-test-stage1-lora", row_in2)
    check("adapter with base present derives a SERVEABLE row",
          derived2 is not None and derived2.get("serveable") is True)
    check("serveable adapter row carries no unserveable reason",
          "unserveable_reason" not in derived2)
    check("serveable adapter row has no unserveable tasks",
          derived2.get("unserveable_tasks") == [])

    # A plain non-adapter row is unaffected by any of this.
    derived3, why3 = mc.derive_model_config_row("Qwen3-Plain", {
        "name": "Qwen3-Plain", "hub_id": "Qwen/Qwen3-Plain", "framework": "transformers",
        "tasks": ["text-generation"], "primary_task": "text-generation", "dir": plain,
    })
    check("ordinary row still derives serveable with no adapter fields",
          derived3 is not None and derived3.get("serveable") is True
          and derived3.get("base_model") is None
          and "unserveable_reason" not in derived3)


    print(f"\nALL {ok} CHECKS PASSED")
