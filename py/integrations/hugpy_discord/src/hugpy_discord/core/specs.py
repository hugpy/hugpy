"""chatshare CommandSpecs mirroring the Discord slash commands.

One entry per command the chatshare adapter exposes, with the SAME names,
sub-command form (``"model set"``), option names/types/ranges/choices, and
``autocomplete: true`` where the Discord side uses ``model_autocomplete``.
Attachment options map to ``type: "attachment"``. ``download``/``canceljob`` are
``admin_only`` (the chatshare server enforces it against site admins).

Discord-only surfaces are intentionally absent: ``/link`` (Discord→hugpy principal
link), the escalation buttons, bridged channels, the console relay, and the
member/channel reporters. See the module ``KNOWN_DROPPED`` note.
"""
from __future__ import annotations

from hugpy_discord.core.commands import (
    KEYWORD_PRESETS, SUMMARY_MODES, TASKS, WHISPER_SIZES)

# Surfaces deliberately NOT mirrored to chatshare (Discord-only), for the record.
KNOWN_DROPPED = (
    "link",                  # Discord account → hugpy principal token link
    "escalation-buttons",    # operator-gated keeper escalation UI
    "bridged-channels",      # console session bridge / console relay
    "member/channel-reporters",
)


def _opt(name, type, description, *, required=False, choices=None, min=None,
         max=None, max_length=None, autocomplete=False) -> dict:
    o: dict = {"name": name, "type": type, "description": description}
    if required:
        o["required"] = True
    if choices is not None:
        o["choices"] = [{"name": c, "value": c} for c in choices]
    if min is not None:
        o["min"] = min
    if max is not None:
        o["max"] = max
    if max_length is not None:
        o["max_length"] = max_length
    if autocomplete:
        o["autocomplete"] = True
    return o


def _model_opt(description="Model override") -> dict:
    return _opt("model", "string", description, autocomplete=True)


def command_specs() -> list[dict]:
    """The full CommandSpec list for ``commands.set`` (order = palette order)."""
    return [
        {"name": "chat", "description": "Chat with the hugpy model", "options": [
            _opt("prompt", "string", "What to say", required=True),
            _opt("model", "string",
                 "Model to use for this turn (defaults to your /model choice)",
                 autocomplete=True),
            _opt("attachment", "attachment",
                 "Optional file (image/audio/document) to include"),
            _opt("private", "boolean", "Only you see the reply"),
            _opt("temperature", "number",
                 "Sampling temperature, 0-2 (central default: 0.1)", min=0.0, max=2.0),
            _opt("top_p", "number",
                 "Nucleus sampling, 0-1 (central default: 1.0)", min=0.0, max=1.0),
            _opt("max_tokens", "integer",
                 "Cap the response length in tokens (default: unbounded)",
                 min=1, max=32768),
            _opt("do_sample", "boolean",
                 "Enable stochastic sampling (central default: off)"),
        ]},
        {"name": "reset",
         "description": "Forget this room's conversation history", "options": []},
        {"name": "running",
         "description": "List in-progress generations", "options": []},
        {"name": "stop", "description": "Stop in-progress generation(s)", "options": [
            _opt("turn_id", "string",
                 "A specific generation from /running "
                 "(default: everything in this room)"),
        ]},
        {"name": "model set", "description": "Set your default model", "options": [
            _opt("model", "string", "Model to set as your default",
                 required=True, autocomplete=True),
        ]},
        {"name": "model show",
         "description": "Show your current default model", "options": []},
        {"name": "model clear",
         "description": "Clear your default model", "options": []},
        {"name": "embed",
         "description": "Embed text into vectors (feature-extraction)", "options": [
             _opt("text", "string",
                  "Text to embed — separate multiple texts with '||' or newlines",
                  required=True),
             _opt("normalize", "boolean",
                  "L2-normalize the vectors (central default: on)"),
             _opt("batch_size", "integer",
                  "Encoder batch size (central default: 32)", min=1, max=256),
             _model_opt("Model override (default: central's embedder)"),
         ]},
        {"name": "similarity",
         "description": "Rank candidates by semantic similarity", "options": [
             _opt("text", "string", "Query text", required=True),
             _opt("compare_to", "string",
                  "Candidates — separate with '||' or newlines", required=True),
             _opt("normalize", "boolean",
                  "L2-normalize before comparing (central default: on)"),
             _model_opt("Model override (default: central's embedder)"),
         ]},
        {"name": "imagine",
         "description": "Generate an image from text", "options": [
             _opt("prompt", "string", "What to generate", required=True),
             _opt("negative", "string", "What to avoid in the image"),
             _opt("width", "integer",
                  "Image width in px (multiple of 8; default: model's native)",
                  min=64, max=4096),
             _opt("height", "integer",
                  "Image height in px (multiple of 8; default: model's native)",
                  min=64, max=4096),
             _opt("steps", "integer",
                  "Inference steps (default: model's native)", min=1, max=200),
             _opt("guidance", "number",
                  "Guidance scale, 0-50 (default: model's native)", min=0.0, max=50.0),
             _opt("seed", "integer", "Seed for reproducible output"),
             _opt("count", "integer", "How many images, 1-4 (default: 1)",
                  min=1, max=4),
             _model_opt("Model override (default: central's image model)"),
         ]},
        {"name": "task",
         "description": "Run any hugpy task with explicit parameters", "options": [
             _opt("task", "string", "Dispatch task key", required=True, choices=TASKS),
             _opt("input", "string",
                  "Primary input (prompt/text; '||'-separated for embed tasks)"),
             _opt("attachment", "attachment", "File input (image/audio/document)"),
             _model_opt(),
             _opt("params", "string",
                  'Extra execute_prompt kwargs as JSON, e.g. {"seed": 7}'),
             _opt("temperature", "number", "Sampling temperature", min=0.0, max=2.0),
             _opt("top_p", "number", "Nucleus sampling", min=0.0, max=1.0),
             _opt("max_tokens", "integer", "Token cap", min=1, max=32768),
             _opt("do_sample", "boolean", "Enable stochastic sampling"),
         ]},
        {"name": "tasks",
         "description": "List hugpy task categories and their default models",
         "options": []},
        {"name": "status",
         "description": "hugpy central health, serving models, workers",
         "options": []},
        {"name": "models",
         "description": "List models in the hugpy registry", "options": [
             _opt("installed_only", "boolean", "Only show installed models"),
         ]},
        {"name": "download", "description": "Download a model (registry key or HF hub id)",
         "admin_only": True, "options": [
             _opt("model", "string", "Registry model key, or a hub id like org/name",
                  required=True, autocomplete=True),
         ]},
        {"name": "jobs", "description": "List download jobs", "options": []},
        {"name": "canceljob", "description": "Cancel a download job",
         "admin_only": True, "options": [
             _opt("job_id", "string", "The download job id", required=True),
         ]},
        {"name": "hf", "description": "Search the Hugging Face hub", "options": [
            _opt("query", "string", "Search terms", required=True),
            _opt("task", "string", "Pipeline tag filter (e.g. text-generation)"),
        ]},
        {"name": "summarize",
         "description": "Summarize text or a file (dedicated summarizer)", "options": [
             _opt("text", "string", "Text to summarize"),
             _opt("attachment", "attachment", "Or attach a document"),
             _opt("mode", "string", "Summary length (default: central decides)",
                  choices=SUMMARY_MODES),
             _opt("preset", "string", "Named parameter bundle: short/medium/long",
                  choices=("short", "medium", "long")),
             _model_opt("Model override (default: central's summarizer)"),
         ]},
        {"name": "keywords",
         "description": "Extract keywords (KeyBERT + spaCy)", "options": [
             _opt("text", "string", "Text to extract keywords from"),
             _opt("preset", "string", "Keyword preset (default: seo)",
                  choices=KEYWORD_PRESETS),
             _opt("top_n", "integer",
                  "How many keywords to consider (default: preset's)", min=1, max=100),
             _opt("diversity", "number",
                  "Result diversity, 0-1 (default: preset's)", min=0.0, max=1.0),
             _opt("attachment", "attachment", "Or attach a document"),
             _model_opt("Embedding model override (default: central's)"),
         ]},
        {"name": "transcribe",
         "description": "Transcribe attached audio/video (whisper)", "options": [
             _opt("attachment", "attachment", "Audio or video file", required=True),
             _opt("language", "string", "Source language hint (default: english)"),
             _opt("size", "string", "Whisper model size (default: central decides)",
                  choices=WHISPER_SIZES),
             _opt("translate", "boolean", "Translate to English instead of transcribing"),
             _opt("timestamps", "boolean", "Include per-segment timestamps"),
             _model_opt("Model override (default: central's whisper)"),
         ]},
        {"name": "describe",
         "description": "Analyze an attached image (vision model)", "options": [
             _opt("attachment", "attachment", "Image to analyze", required=True),
             _opt("prompt", "string",
                  "What to ask about it (default: describe in detail)"),
             _opt("max_tokens", "integer",
                  "Cap the response length in tokens", min=1, max=32768),
             _model_opt("Model override (default: central's vision model)"),
         ]},
    ]
