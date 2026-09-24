"""A video diffusion model must not advertise itself as a chat model.

Operator, 2026-07-27, seeing the console: the Wan rows showed `ctx 32768`, a
`4-bit` bitsandbytes tick, an LLM VRAM price, and `text-generation` — for
text-to-video models. _"those are all 4-bit enabled."_

ROOT CAUSE. `_base_tasks` walks `pipeline_tag` → `model_type` → the four known
sets, then falls off the end at:

    return ["text-generation"]      # conservative floor

Every Wan row reached it with NO `pipeline_tag` and NO `model_type`, so all seven
advertised `["text-generation"]`. Downstream believed it: `bnb_available` becomes
true for a non-quantized transformers row (the 4-bit lever), the LLM allocator
prices it, and it is eligible for chat routing.

THE TRUTH WAS IN THE FILE. `Wan2.1-T2V-1.3B/config.json`:

    {"_class_name": "WanModel", "_diffusers_version": "0.30.0",
     "model_type": "t2v", "dim": 1536, "ffn_dim": 8960, "num_layers": 30}

The model names itself, and `_base_tasks` already reads `model_type` — the
classifier just had no video vocabulary, and the row never carried the field.

TWO FIXES, both asserted here:
  1. `_VIDEO_T2V` / `_VIDEO_I2V` model_type sets → `text-to-video` /
     `image-to-video` (added to HF_TASK_TO_TASKS). 2026-09-24: a FULL video
     pipeline is now SERVEABLE — its runner lives in hugpy_video's studio
     registry, so ("transformers","text-to-video") / ("transformers",
     "image-to-video") are RUNNER_PAIRS and the row is graded like any other
     model (no-grader until a video suite exists) instead of the blanket
     "unservable/pipeline component". Neither video task is in TASK_DEFAULTS, so
     no chat/media route can bind a video model. A single-file GGUF video
     transformer stays a pipeline-component (it is not a full pipeline).
  2. `_enrich_model_type` fills `model_type` from the model's own `config.json`
     when the row lacks it — content-authoritative, the same discipline
     `_correct_gguf_vision` applies to the mmproj question.

Run: venv/bin/python -m pytest tests/test_video_model_classification.py -q
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_engine.config.models import models_config as M


def _wan_dir(tmp_path, model_type="t2v"):
    d = tmp_path / "Wan2.1-T2V-1.3B"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "_class_name": "WanModel",
        "_diffusers_version": "0.30.0",
        "model_type": model_type,
        "dim": 1536, "ffn_dim": 8960, "num_layers": 30,
    }))
    return str(d)


def test_a_t2v_model_is_not_a_chat_model(tmp_path):
    """THE BUG. Without enrichment the row has no task of its own.

    (k61, 2026-07-31: the floor this precondition used to assert —
    ``["text-generation"]`` for a row that says NOTHING about itself — is gone.
    A no-signal row is now ``needs-classification``: it refuses with a remedy
    instead of promising to be a chat model. The point of THIS test is unchanged
    — enrichment is what makes the Wan row classify correctly.)"""
    row = {"hub_id": "Wan-AI/Wan2.1-T2V-1.3B", "dir": _wan_dir(tmp_path)}
    assert M._derive_tasks("transformers", row) == ["needs-classification"], (
        "precondition: a row with no model_type has nothing to classify on")
    enriched = M._enrich_model_type("transformers", row)
    assert enriched["model_type"] == "t2v"
    assert M._derive_tasks("transformers", enriched) == ["text-to-video"]


def test_vace_classifies_as_image_to_video(tmp_path):
    row = {"hub_id": "Wan-AI/Wan2.1-VACE-1.3B",
           "dir": _wan_dir(tmp_path, model_type="vace")}
    enriched = M._enrich_model_type("transformers", row)
    assert M._derive_tasks("transformers", enriched) == ["image-to-video"]


def test_full_video_pipeline_is_serveable_via_hugpy_video(tmp_path):
    """2026-09-24: a FULL video pipeline is serveable — its runner lives in
    hugpy_video's studio registry, so ("transformers","text-to-video") /
    ("transformers","image-to-video") are RUNNER_PAIRS. It is graded like any
    other model (no video suite yet -> no-grader) instead of the blanket
    "unservable/pipeline component" it read as before. A GGUF video task stays
    OUT of RUNNER_PAIRS: a single-file GGUF is a component, never a full pipeline
    (and no chat/media default binds a video model — neither is in TASK_DEFAULTS).
    """
    from hugpy_engine.categories import RUNNER_PAIRS, TASK_DEFAULTS, MEDIA_DEFAULTS
    for task in ("text-to-video", "image-to-video"):
        assert ("transformers", task) in RUNNER_PAIRS, task
        assert ("gguf", task) not in RUNNER_PAIRS, task
        assert task not in TASK_DEFAULTS, task
    assert MEDIA_DEFAULTS["video"] == MEDIA_DEFAULTS["audio"]   # video routes to ASR, not a t2v model
    # A discovered t2v row is written serveable=True with no unserveable_reason.
    cfg, reason = M.derive_model_config_row(
        "Wan2.1-T2V-1.3B",
        {"hub_id": "Wan-AI/Wan2.1-T2V-1.3B", "framework": "transformers",
         "pipeline_tag": "text-to-video", "dir": _wan_dir(tmp_path)})
    assert cfg is not None, reason
    assert cfg["tasks"] == ["text-to-video"] and cfg["serveable"] is True
    assert cfg.get("unserveable_reason") in (None,) and cfg["unserveable_tasks"] == []


def test_an_explicit_row_value_always_wins(tmp_path):
    """Enrichment fills a GAP; it never overrides what the row already states."""
    row = {"model_type": "llama", "dir": _wan_dir(tmp_path)}
    assert M._enrich_model_type("transformers", row)["model_type"] == "llama"


def test_ordinary_llms_are_untouched():
    """The blast radius must be zero for every non-video model."""
    assert M._derive_tasks("transformers", {"model_type": "llama"}) == ["text-generation"]
    assert M._derive_tasks("transformers", {"model_type": "qwen2"}) == ["text-generation"]
    assert M._derive_tasks("transformers", {"model_type": "whisper"}) == [
        "automatic-speech-recognition"]
    assert "image-text-to-text" in M._derive_tasks(
        "transformers", {"model_type": "qwen2_5_vl"})


def test_gguf_rows_are_not_touched(tmp_path):
    """A GGUF's type comes from its own header and _base_tasks handles that branch
    before model_type is consulted — enrichment must not interfere."""
    row = {"dir": _wan_dir(tmp_path)}
    assert M._enrich_model_type("gguf", row) is row, "gguf rows pass through"


def test_enrichment_never_raises_on_a_bad_config(tmp_path):
    """Discovery walks whatever is on disk. A missing / unreadable / malformed
    config.json must degrade, never break the row."""
    empty = tmp_path / "no_config"
    empty.mkdir()
    assert M._enrich_model_type("transformers", {"dir": str(empty)}).get("model_type") is None

    bad = tmp_path / "bad_config"
    bad.mkdir()
    (bad / "config.json").write_text("{not json")
    assert M._enrich_model_type("transformers", {"dir": str(bad)}).get("model_type") is None

    # no dir at all
    assert M._enrich_model_type("transformers", {}).get("model_type") is None

    # config.json present but model_type absent / not a string
    odd = tmp_path / "odd"
    odd.mkdir()
    (odd / "config.json").write_text(json.dumps({"model_type": 17}))
    assert M._enrich_model_type("transformers", {"dir": str(odd)}).get("model_type") is None


def test_the_real_wan_config_on_disk_classifies(tmp_path):
    """Against the ACTUAL weights on this fleet, when present."""
    real = "/mnt/llm_storage/models/transformers/Wan-AI/Wan2.1-T2V-1.3B"
    if not os.path.isfile(os.path.join(real, "config.json")):
        return  # weights not on this box — the synthetic cases above still cover it
    enriched = M._enrich_model_type("transformers", {"dir": real})
    assert enriched.get("model_type") == "t2v", enriched.get("model_type")
    assert M._derive_tasks("transformers", enriched) == ["text-to-video"]


def test_the_diffusers_repack_shape_is_also_detected(tmp_path):
    """BOTH shapes exist side by side on this fleet and must both classify:
      * Wan2.1-VACE-1.3B            -> config.json {"model_type": "vace"}
      * Wan2.1-VACE-1.3B-diffusers  -> model_index.json {"_class_name": "WanVACEPipeline"}
    Reading only config.json left the repack advertising text-generation.
    """
    d = tmp_path / "repack"
    d.mkdir()
    (d / "model_index.json").write_text(json.dumps({
        "_class_name": "WanVACEPipeline", "_diffusers_version": "0.34.0.dev0"}))
    row = {"dir": str(d), "tasks": ["text-generation"]}
    assert M._enrich_model_type("transformers", row)["model_type"] == "vace"
    assert M._correct_video_task("transformers", ["text-generation"], row) == ["image-to-video"]


def test_a_text_pipeline_repack_is_not_called_video(tmp_path):
    """The _class_name branch fires ONLY for Wan pipelines. A diffusers repack of
    something else must not be swept up."""
    d = tmp_path / "sd"
    d.mkdir()
    (d / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionPipeline"}))
    row = {"dir": str(d), "tasks": ["text-to-image"]}
    assert M._enrich_model_type("transformers", row).get("model_type") is None
    assert M._correct_video_task("transformers", ["text-to-image"], row) == ["text-to-image"]


def test_the_corrector_only_overrides_text_generation(tmp_path):
    """It corrects the WRONG floor value. A row already carrying a real task list
    (e.g. text-to-image) is never rewritten, even for a Wan dir."""
    row = {"dir": _wan_dir(tmp_path), "tasks": ["text-to-image"]}
    assert M._correct_video_task("transformers", ["text-to-image"], row) == ["text-to-image"]


# ── pipeline-component GGUFs (LTX text-encoder split) ────────────────────────
# A diffusion/video pipeline splits into sub-trees; its text encoder is often a
# real LLM architecture (LTX-2 ships a Gemma-3 encoder) and CANNOT be told from
# a chat model by its header — only by its path. LTX-2.3-uncensored-fp8 was
# routed to the llama chat path and rejected at load because its registered file
# is split/text_encoders/gemma-3-12b-it-...gguf (fleet bench 2026-07-30).

def test_ltx_text_encoder_split_is_a_pipeline_component():
    row = {"filename": "split/text_encoders/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf"}
    assert M._correct_pipeline_component("gguf", ["text-generation"], row) == ["pipeline-component"]


def test_vae_split_is_a_pipeline_component():
    row = {"filename": "split/vae/diffusion_pytorch_model.gguf"}
    assert M._correct_pipeline_component("gguf", ["text-generation"], row) == ["pipeline-component"]


def test_standalone_gemma_gguf_is_left_a_chat_model():
    """The whole point of the path guard: a real Gemma/Qwen GGUF at its model
    root is NOT under a component segment, so it stays servable."""
    row = {"filename": "gemma-3-12b-it-Q4_K_M.gguf"}
    assert M._correct_pipeline_component("gguf", ["text-generation"], row) == ["text-generation"]


def test_component_word_in_the_name_only_does_not_trip_it():
    """A model whose FILENAME contains 'encoder' but has no component PATH
    segment must not be swept up — the signal is the directory, not the word."""
    row = {"filename": "my-text-encoder-tuned-Q4_K_M.gguf"}
    assert M._correct_pipeline_component("gguf", ["text-generation"], row) == ["text-generation"]


def test_pipeline_component_guard_is_gguf_only():
    row = {"filename": "unet/text_encoder/model.safetensors"}
    assert M._correct_pipeline_component("transformers", ["text-generation"], row) == ["text-generation"]


# ── single-file video-model GGUFs (Wan VACE) ─────────────────────────────────
# QuantStack/Wan2.1_14B_VACE-GGUF ships one file at the model root
# (Wan2.1_14B_VACE-Q8_0.gguf): the quantized diffusion transformer ALONE, no VAE
# / text-encoder / scheduler, so it is a pipeline-component like the split fp8
# checkpoints. It has no component PATH segment and no config model_type, so the
# only truthful signal is the Hub card's pipeline_tag / tags — which the gguf
# floor ignores, flooring it to text-generation (then even graded weak).

def test_single_file_video_gguf_is_a_pipeline_component_by_pipeline_tag():
    row = {"filename": "Wan2.1_14B_VACE-Q8_0.gguf", "pipeline_tag": "text-to-video"}
    assert M._correct_pipeline_component("gguf", ["text-generation"], row) == ["pipeline-component"]


def test_single_file_video_gguf_is_a_pipeline_component_by_tags():
    row = {"filename": "Wan2.1-VACE-1.3B-F16.gguf",
           "tags": ["gguf", "video", "video-generation", "text-to-video"]}
    assert M._correct_pipeline_component("gguf", ["text-generation"], row) == ["pipeline-component"]


def test_plain_chat_gguf_with_text_generation_tag_is_left_alone():
    """A real chat GGUF carries pipeline_tag text-generation and no video tag, so
    the video-gguf branch never fires."""
    row = {"filename": "gemma-3-12b-it-Q4_K_M.gguf", "pipeline_tag": "text-generation",
           "tags": ["gguf", "text-generation", "conversational"]}
    assert M._correct_pipeline_component("gguf", ["text-generation"], row) == ["text-generation"]


# ── latent upscaler / upsampler pipelines are components ─────────────────────
# ltxv-spatial-upscaler-0.9.7 / LTX-Video-spatial-upscaler-0.9.8: diffusers dirs
# whose model_index _class_name is LTXLatentUpsamplePipeline and whose
# pipeline_tag is video-to-video (not in HF_TASK_TO_TASKS) — floored to
# text-generation, i.e. a video upscaler offering itself as a chat model.

def test_latent_upscaler_pipeline_class_is_a_component():
    from hugpy_engine.model_classifier import tasks_for_pipeline_class, PIPELINE_COMPONENT_TASK
    assert tasks_for_pipeline_class("LTXLatentUpsamplePipeline") == [PIPELINE_COMPONENT_TASK]
    # a StableDiffusion upscale pipeline is also a component (no upscale runner)
    assert tasks_for_pipeline_class("StableDiffusionLatentUpscalePipeline") == [PIPELINE_COMPONENT_TASK]


def test_upscaler_dir_classifies_as_component_through_correct_diffusers_task(tmp_path):
    d = tmp_path / "ltxv-spatial-upscaler"
    d.mkdir()
    (d / "model_index.json").write_text(json.dumps({"_class_name": "LTXLatentUpsamplePipeline"}))
    row = {"dir": str(d), "tasks": ["text-generation"]}
    assert M._correct_diffusers_task("transformers", ["text-generation"], row) == ["pipeline-component"]


def test_real_image_generator_is_untouched_by_the_upscaler_rule():
    """The blast radius is upscalers only: a normal generative pipeline still
    gets text-to-image + image-to-image."""
    from hugpy_engine.model_classifier import tasks_for_pipeline_class
    assert tasks_for_pipeline_class("StableDiffusionXLPipeline") == ["text-to-image", "image-to-image"]
    # video generator pipelines still defer to the video corrector (None here)
    assert tasks_for_pipeline_class("WanVACEPipeline") is None
