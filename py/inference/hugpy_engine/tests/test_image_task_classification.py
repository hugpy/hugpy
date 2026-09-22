"""An image model's task comes from its own directory — never from a guess.

THE INCIDENT (operator, 2026-07-31: "flux2 never fires", 7 failed attempts).
Every flux2 key was task-misclassified, and the wrong task was stamped into all
three task stores (central discovery, worker discovery, the per-model hugpy.json
sidecar — the sovereign one):

  * FLUX.2-klein-base-9B-bucket-uncensored — a COMPLETE diffusers pipeline whose
    own model_index.json says {"_class_name": "Flux2KleinPipeline"} — was stamped
    tasks: ["image-to-image"] only, so every text-to-image call refused.
  * Flux-Uncensored-V2 — a LoRA-only dir (lora.safetensors, no model_index.json)
    — was left tasks: null, and null silently defaulted to text-generation, so an
    image LoRA was offered as an LLM and refused every image call with a nonsense
    "supported: ['text-generation']".
  * flux2-klein-9b-uncensored-text-encoder — a component (a Qwen3 encoder GGUF),
    correctly text-generation-shaped, but selectable in image UIs where it can
    never serve.

The keeper fixed the DATA by hand; these tests pin the STAMPER so it can't
regress. Everything here runs OFFLINE from fixture dir contents — hub metadata is
a bonus, never a dependency.

Run: venv/bin/python -m pytest tests/test_image_task_classification.py -q
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_engine import model_classifier as C
from hugpy_engine.config.models import models_config as M


# ── fixture dirs — the three shapes the incident named ──────────────────────
def _pipeline_dir(tmp_path, class_name="Flux2KleinPipeline", name="flux2-klein"):
    """A COMPLETE diffusers pipeline: model_index.json naming its class and the
    components it loads."""
    d = tmp_path / name
    d.mkdir()
    (d / "model_index.json").write_text(json.dumps({
        "_class_name": class_name,
        "_diffusers_version": "0.35.0",
        "text_encoder": ["transformers", "Qwen3Model"],
        "tokenizer": ["transformers", "Qwen2Tokenizer"],
        "transformer": ["diffusers", "Flux2Transformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    }))
    (d / "model.safetensors").write_text("weights")
    return str(d)


def _lora_dir(tmp_path, filename="lora.safetensors", name="Flux-Uncensored-V2"):
    """A LoRA-only dir: one adapter file, no model_index.json, no config.json."""
    d = tmp_path / name
    d.mkdir()
    (d / filename).write_text("delta")
    return str(d)


def _encoder_gguf_dir(tmp_path, name="flux2-klein-9b-uncensored-text-encoder"):
    """A pipeline COMPONENT shipped as a GGUF — a real Qwen3 encoder."""
    d = tmp_path / name
    d.mkdir()
    (d / "qwen3-8b-encoder-Q4_K_M.gguf").write_text("gguf")
    return str(d)


# ── 1. model_index.json is authoritative for diffusers dirs ─────────────────
def test_a_complete_pipeline_serves_both_image_directions(tmp_path):
    """THE BUG. Flux2KleinPipeline is a full t2i pipeline; it was stamped
    image-to-image ONLY, so text-to-image refused."""
    verdict = C.classify_model_dir(_pipeline_dir(tmp_path))
    assert verdict["tasks"] == ["text-to-image", "image-to-image"]
    assert verdict["primary_task"] == "text-to-image"
    assert verdict["source"] == "model_index"
    assert verdict["pipeline_class"] == "Flux2KleinPipeline"


def test_the_declaration_beats_the_wrong_stamp(tmp_path):
    """The whole point: the stamp said image-to-image only, in all three stores.
    The pipeline's own declaration OVERRIDES it — otherwise the hand-corrected
    fleet data regresses on the next walk."""
    row = {"dir": _pipeline_dir(tmp_path), "tasks": ["image-to-image"],
           "primary_task": "image-to-image", "name": "FLUX.2-klein-base-9B"}
    assert M._derive_tasks("transformers", row) == ["image-to-image"], (
        "precondition: _base_tasks short-circuits on the stored (wrong) value")
    assert M._correct_diffusers_task("transformers", ["image-to-image"], row) == [
        "text-to-image", "image-to-image"]


def test_an_img2img_only_pipeline_stays_img2img(tmp_path):
    """Not everything widens: an edit pipeline has no text-only path."""
    d = _pipeline_dir(tmp_path, class_name="QwenImageEditPipeline", name="qwen-edit")
    assert C.classify_model_dir(d)["tasks"] == ["image-to-image"]


def test_an_inpaint_variant_adds_inpainting(tmp_path):
    d = _pipeline_dir(tmp_path, class_name="FluxInpaintPipeline", name="flux-inpaint")
    assert C.classify_model_dir(d)["tasks"] == ["image-to-image", "image-inpainting"]


def test_an_unknown_pipeline_class_falls_back_on_its_shape(tmp_path):
    """The map is small on purpose — a pipeline nobody enumerated still
    classifies, from the suffix + the components it declares."""
    d = _pipeline_dir(tmp_path, class_name="SomeBrandNewImagePipeline", name="brandnew")
    assert C.classify_model_dir(d)["tasks"] == ["text-to-image", "image-to-image"]


def test_video_pipelines_are_left_to_the_video_corrector(tmp_path):
    """WanVACEPipeline is a pipeline declaration too, but the video vocabulary is
    derived elsewhere from the same file. Classifying it here would fight
    _correct_video_task — so this module says nothing about it."""
    d = _pipeline_dir(tmp_path, class_name="WanVACEPipeline", name="wan-repack")
    assert C.classify_model_dir(d) == {}
    row = {"dir": d, "tasks": ["text-generation"]}
    assert M._correct_video_task("transformers", ["text-generation"], row) == ["image-to-video"]
    assert M._correct_diffusers_task("transformers", ["image-to-video"], row) == ["image-to-video"]


def test_classification_never_raises_on_a_bad_dir(tmp_path):
    empty = tmp_path / "empty"; empty.mkdir()
    assert C.classify_model_dir(str(empty)) == {}
    bad = tmp_path / "bad"; bad.mkdir()
    (bad / "model_index.json").write_text("{not json")
    assert C.classify_model_dir(str(bad)) == {}
    assert C.classify_model_dir(None) == {}
    assert C.classify_model_dir(str(tmp_path / "nope")) == {}


# ── 2. adapter-only dirs are ADAPTERS, never null ───────────────────────────
def test_a_lora_only_dir_is_an_adapter(tmp_path):
    verdict = C.classify_model_dir(_lora_dir(tmp_path))
    assert verdict["adapter"] is True
    assert verdict["tasks"] == [C.ADAPTER_TASK]
    assert verdict["tasks"] != [] and verdict["tasks"] is not None


def test_the_diffusers_lora_filename_shape_is_detected(tmp_path):
    """Diffusers LoRAs ship NO adapter_config.json — the filename is the only
    signal, which is exactly why Flux-Uncensored-V2 read as 'no information'."""
    d = _lora_dir(tmp_path, filename="pytorch_lora_weights.safetensors", name="difflora")
    assert C.is_adapter_only_dir(d) is True


def test_a_peft_adapter_dir_is_detected(tmp_path):
    d = tmp_path / "peft"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "base_model_name_or_path": "meta-llama/Llama-3.2-3B"}))
    (d / "adapter_model.safetensors").write_text("delta")
    assert C.is_adapter_only_dir(str(d)) is True


def test_a_real_model_is_never_called_an_adapter(tmp_path):
    """Blast radius zero: a standalone model stands alone even if a stray adapter
    config rode along with it."""
    d = tmp_path / "merged"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (d / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
    (d / "model.safetensors").write_text("weights")
    assert C.is_adapter_only_dir(str(d)) is False
    assert C.is_adapter_only_dir(_pipeline_dir(tmp_path)) is False


def test_the_adapter_row_is_kept_visible_and_refuses_by_name(tmp_path):
    """Operator doctrine: silent unavailability is the defect. The row stays on
    the models tab, unserveable, with a reason that names the fix."""
    row = {"hub_id": "enhanceaiteam/Flux-Uncensored-V2",
           "dir": _lora_dir(tmp_path), "tasks": None}
    cfg, why = M.derive_model_config_row("Flux-Uncensored-V2", row)
    assert cfg is not None, why
    assert cfg["tasks"] == [C.ADAPTER_TASK]
    assert cfg["adapter"] is True
    assert cfg["serveable"] is False
    assert "LoRA" in cfg["unserveable_reason"]
    assert "FIX:" in cfg["unserveable_reason"]
    assert "text-generation" not in cfg["tasks"]


# ── 3. null never defaults to text-generation ───────────────────────────────
def test_a_row_with_no_signal_is_unclassified_not_a_chat_model(tmp_path):
    """THE SECOND BUG. `return ["text-generation"]` was unconditional, so a row
    that said nothing about itself was PROMISED to be an LLM."""
    assert M._base_tasks("transformers", {}) == [C.NEEDS_CLASSIFICATION_TASK]


def test_the_floor_still_holds_for_real_transformers_models():
    """Blast radius: a model that DOES identify itself still gets the floor."""
    assert M._base_tasks("transformers", {"model_type": "llama"}) == ["text-generation"]
    assert M._base_tasks("transformers", {"architectures": ["Qwen3ForCausalLM"]}) == [
        "text-generation"]
    assert M._base_tasks("gguf", {}) == ["text-generation"]


def test_an_unclassified_row_refuses_with_the_remedy_named(tmp_path):
    d = tmp_path / "mystery"
    d.mkdir()
    (d / "weights.pt").write_text("?")
    cfg, why = M.derive_model_config_row("mystery", {"hub_id": "o/mystery", "dir": str(d)})
    assert cfg is not None, why
    assert cfg["tasks"] == [C.NEEDS_CLASSIFICATION_TASK]
    assert cfg["serveable"] is False
    reason = cfg["unserveable_reason"]
    assert "unclassified" in reason
    assert "reclassify-images" in reason, "the refusal must NAME the remedy"


def test_no_runner_exists_for_the_non_servable_tokens():
    """These tokens are honest labels, not routes. If someone ever wires a runner
    for one, chat routing could bind an adapter — fail loudly here."""
    from hugpy_engine.categories import RUNNER_PAIRS
    for task in (C.ADAPTER_TASK, C.NEEDS_CLASSIFICATION_TASK):
        for fw in ("transformers", "gguf", "comfy"):
            assert (fw, task) not in RUNNER_PAIRS, (fw, task)


def test_the_resolver_refuses_an_unclassified_model_by_name():
    """The read site. Before: 'supported: ['text-generation']' — a promise the
    model could not keep, naming neither cause nor fix."""
    from hugpy_engine.resolvers.model_resolver import _refuse_if_unclassified

    class _Cfg:
        tasks = [C.NEEDS_CLASSIFICATION_TASK]
        base_model = None

    try:
        _refuse_if_unclassified("mystery", _Cfg())
    except ValueError as exc:
        assert "unclassified" in str(exc) and "FIX:" in str(exc)
    else:
        raise AssertionError("an unclassified model must refuse, not resolve")

    class _Adapter:
        tasks = [C.ADAPTER_TASK]
        base_model = "black-forest-labs/FLUX.2-klein"

    try:
        _refuse_if_unclassified("Flux-Uncensored-V2", _Adapter())
    except ValueError as exc:
        assert "adapter" in str(exc).lower() and "FIX:" in str(exc)
    else:
        raise AssertionError("an adapter must refuse, not resolve")


# ── 4. capability-matching pickers ──────────────────────────────────────────
def test_an_encoder_component_never_advertises_an_image_task(tmp_path):
    """The third row of the incident: a text-encoder GGUF is text-generation-
    shaped and correct — it just must never look like an image model. Its tasks
    carry no image capability, so a capability-filtered picker excludes it."""
    row = {"hub_id": "o/flux2-klein-9b-uncensored-text-encoder",
           "dir": _encoder_gguf_dir(tmp_path),
           "filename": "split/text_encoders/qwen3-8b-encoder-Q4_K_M.gguf"}
    cfg, why = M.derive_model_config_row("flux2-encoder", row)
    assert cfg is not None, why
    assert not ({"text-to-image", "image-to-image"} & set(cfg["tasks"]))
    assert cfg["serveable"] is False


def test_a_partially_servable_row_is_still_servable(tmp_path):
    """An inpaint pipeline advertises image-to-image (runner: yes) AND
    image-inpainting (runner: no). It must stay servable for what it CAN do."""
    row = {"hub_id": "o/flux-inpaint",
           "dir": _pipeline_dir(tmp_path, class_name="FluxInpaintPipeline",
                                name="inpaint")}
    cfg, why = M.derive_model_config_row("flux-inpaint", row)
    assert cfg is not None, why
    assert cfg["tasks"] == ["image-to-image", "image-inpainting"]
    assert cfg["serveable"] is True
    assert cfg["unserveable_tasks"] == ["image-inpainting"]


# ── 5. the one-shot re-stamp ────────────────────────────────────────────────
def _discovery_report(tmp_path, rows):
    path = tmp_path / "model_discovery.json"
    path.write_text(json.dumps(rows))
    return str(path)


def test_reclassify_corrects_the_sidecar_and_the_row(tmp_path):
    """The fleet's EXISTING wrong stamps get corrected by code, not by hand."""
    from hugpy_engine.apis.reclassify import reclassify_images
    from hugpy_storage.hugpy_marker import read_hugpy_marker, write_hugpy_marker

    d = _pipeline_dir(tmp_path)
    write_hugpy_marker(d, hub_id="o/FLUX.2-klein", tasks=["image-to-image"],
                       primary_task="image-to-image", framework="transformers")
    report_path = _discovery_report(tmp_path, {
        "FLUX.2-klein": {"dir": d, "hub_id": "o/FLUX.2-klein",
                         "tasks": ["image-to-image"],
                         "primary_task": "image-to-image"}})

    dry = reclassify_images(apply=False, discovery_path=report_path)
    assert len(dry["changed"]) == 1
    assert dry["changed"][0]["to"] == ["text-to-image", "image-to-image"]
    assert read_hugpy_marker(d)["tasks"] == ["image-to-image"], "dry run writes nothing"

    applied = reclassify_images(apply=True, discovery_path=report_path)
    assert applied["changed"][0]["applied"] is True
    assert read_hugpy_marker(d)["tasks"] == ["text-to-image", "image-to-image"]
    assert read_hugpy_marker(d)["primary_task"] == "text-to-image"
    assert read_hugpy_marker(d)["hub_id"] == "o/FLUX.2-klein", "identity preserved"
    rows = json.loads(Path(report_path).read_text())
    assert rows["FLUX.2-klein"]["tasks"] == ["text-to-image", "image-to-image"]


def test_reclassify_is_idempotent(tmp_path):
    """A second run finds nothing — it compares before writing."""
    from hugpy_engine.apis.reclassify import reclassify_images
    from hugpy_storage.hugpy_marker import write_hugpy_marker

    d = _pipeline_dir(tmp_path)
    write_hugpy_marker(d, hub_id="o/FLUX.2-klein", tasks=["image-to-image"],
                       primary_task="image-to-image", framework="transformers")
    report_path = _discovery_report(tmp_path, {"FLUX.2-klein": {"dir": d}})

    assert len(reclassify_images(apply=True, discovery_path=report_path)["changed"]) == 1
    second = reclassify_images(apply=True, discovery_path=report_path)
    assert second["changed"] == [], "re-stamping a correct dir must be a no-op"


def test_reclassify_stamps_an_adapter_and_leaves_llms_alone(tmp_path):
    from hugpy_engine.apis.reclassify import reclassify_images
    from hugpy_storage.hugpy_marker import read_hugpy_marker, write_hugpy_marker

    lora = _lora_dir(tmp_path)
    write_hugpy_marker(lora, hub_id="o/Flux-Uncensored-V2", tasks=None,
                       framework="transformers")
    llm = tmp_path / "qwen"
    llm.mkdir()
    (llm / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (llm / "model.safetensors").write_text("weights")
    write_hugpy_marker(str(llm), hub_id="Qwen/Qwen3-8B", tasks=["text-generation"],
                       primary_task="text-generation", framework="transformers")

    report_path = _discovery_report(tmp_path, {
        "Flux-Uncensored-V2": {"dir": lora}, "Qwen3-8B": {"dir": str(llm)}})
    report = reclassify_images(apply=True, discovery_path=report_path)

    keys = {c["model_key"] for c in report["changed"]}
    assert keys == {"Flux-Uncensored-V2"}, "an LLM is never touched by this sweep"
    assert read_hugpy_marker(lora)["tasks"] == [C.ADAPTER_TASK]
    assert read_hugpy_marker(lora)["adapter"] is True
    assert read_hugpy_marker(str(llm))["tasks"] == ["text-generation"]


# ── 6. the PEFT path k61 must not break ─────────────────────────────────────
def test_a_pairable_peft_adapter_inherits_its_base_task(tmp_path, monkeypatch):
    """An adapter that NAMES its base is a delta the load path can pair (base +
    delta). Removing the text-generation default must not turn those working rows
    into 'unclassified' — the base's own config is READ, and its task inherited."""
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (base / "model.safetensors").write_text("weights")

    d = tmp_path / "stage1-lora"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "base_model_name_or_path": "Qwen/Qwen3.5-0.8B"}))
    (d / "adapter_model.safetensors").write_text("delta")

    # The dir IS adapter-shaped, but it is PAIRABLE — the classifier defers.
    assert C.is_adapter_only_dir(str(d)) is True
    assert C.classify_model_dir(str(d)) == {}

    import hugpy_engine.peft_adapters as P
    monkeypatch.setattr(P, "find_base_model_dir", lambda b, *a, **k: str(base))
    row = {"hub_id": "o/stage1-lora", "dir": str(d),
           "base_model": "Qwen/Qwen3.5-0.8B"}
    assert M._inherit_adapter_base_task(
        "transformers", [C.NEEDS_CLASSIFICATION_TASK], row) == ["text-generation"]


def test_inheritance_never_invents_a_task_for_an_absent_base(tmp_path, monkeypatch):
    import hugpy_engine.peft_adapters as P
    monkeypatch.setattr(P, "find_base_model_dir", lambda b, *a, **k: None)
    row = {"hub_id": "o/lonely-lora", "base_model": "someone/not-here"}
    assert M._inherit_adapter_base_task(
        "transformers", [C.NEEDS_CLASSIFICATION_TASK], row) == [C.NEEDS_CLASSIFICATION_TASK]


def test_inheritance_never_overrides_a_stated_task():
    row = {"base_model": "Qwen/Qwen3.5-0.8B"}
    assert M._inherit_adapter_base_task("transformers", ["text-to-image"], row) == [
        "text-to-image"]


def test_a_checkpoint_whose_NAME_contains_lora_is_not_an_adapter(tmp_path):
    """Found by the real store (2026-07-31 dry run):
    ``anyloracheckpoint-bakedvaeblessedfp16.safetensors`` is a COMPLETE SD
    checkpoint. A substring test for "lora" demoted it to an adapter and killed a
    working text-to-image model — the marker must be a WORD in the filename."""
    d = tmp_path / "anylora"
    d.mkdir()
    (d / "anyloracheckpoint-bakedvaeblessedfp16.safetensors").write_text("weights")
    assert C.is_adapter_only_dir(str(d)) is False
    for real in ("lora.safetensors", "pytorch_lora_weights.safetensors",
                 "flux-lora-v2.safetensors", "adapter_model.safetensors"):
        one = tmp_path / f"is-{real}"
        one.mkdir()
        (one / real).write_text("delta")
        assert C.is_adapter_only_dir(str(one)) is True, real
