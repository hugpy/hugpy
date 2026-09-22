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
     `image-to-video` (added to HF_TASK_TO_TASKS). They are advertised truthfully
     but NOT servable by the LLM plane: there is no ("transformers","text-to-video")
     RUNNER_PAIR, so chat routing excludes them. The studio arm serves them from
     its own registry.
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


def test_video_tasks_have_no_llm_runner():
    """The point of classifying honestly: these become VISIBLE but not
    chat-servable. If someone later adds a ("transformers","text-to-video")
    RUNNER_PAIR, chat routing could bind a video model — fail loudly here."""
    from hugpy_engine.categories import RUNNER_PAIRS
    for task in ("text-to-video", "image-to-video"):
        assert ("transformers", task) not in RUNNER_PAIRS, task
        assert ("gguf", task) not in RUNNER_PAIRS, task


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
