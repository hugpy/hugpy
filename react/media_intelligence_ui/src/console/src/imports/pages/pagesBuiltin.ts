// src/console/src/imports/pages/pagesBuiltin.ts
//
// The LIVE media-intelligence tool pages (imported for side-effect by
// UtilityRoute -> each registerPage runs at module load). These are pointed at
// CURRENT hugpy:
//   - the dedicated /ml/* amenity endpoints (task fixed server-side + reserved
//     ML pool; see flask_app .../routes/ml_routes.py), and
//   - /prompt for plain text generation.
//
// The legacy pre-hugpy.ai routes (/summarizer/*, /keybert/*, /audio|image|pdf|
// video|url/{text,summarize,analyze}) are GONE on current hugpy and have no 1:1
// equivalent (the source-aware "summarize a PDF/URL/video" tools would need a
// client-side extract -> /ml/summarize composition, not built yet), so they're
// removed rather than left to 404. Field names match the dispatch builders:
// text tasks read `text`/`prompt`; whisper reads `file`; vision reads `image_path`.
import { registerPage } from "./pagesRegistry";

registerPage({
  key: "ml/summarize",
  title: "Summarize Text",
  category: "text",
  path: "/ml/summarize",
  method: "POST",
  resultKind: "json",
  accepts: ["text"],
  produces: "summarize",
  fields: [
    { name: "text", label: "Text", kind: "textarea", required: true, source: "text" },
    { name: "max_length", label: "Max length", kind: "number", default: 512 },
  ],
});

registerPage({
  key: "ml/keywords",
  title: "Extract Keywords",
  category: "text",
  path: "/ml/keywords",
  method: "POST",
  resultKind: "json",
  accepts: ["text"],
  produces: "keywords",
  fields: [
    { name: "text", label: "Text", kind: "textarea", required: true, source: "text" },
    {
      name: "preset",
      label: "Preset",
      kind: "select",
      default: "seo",
      choices: ["default", "seo", "metadata", "social", "long_tail"] as const,
    },
  ],
});

registerPage({
  key: "ml/embed",
  title: "Embeddings",
  category: "text",
  path: "/ml/embed",
  method: "POST",
  resultKind: "json",
  accepts: ["text"],
  produces: "metadata",
  fields: [
    { name: "text", label: "Text", kind: "textarea", required: true, source: "text" },
  ],
});

registerPage({
  key: "ml/imagine",
  title: "Text to Image",
  category: "image",
  path: "/ml/imagine",
  method: "POST",
  resultKind: "json",
  accepts: ["text"],
  produces: "imagegen",
  fields: [
    { name: "prompt", label: "Prompt", kind: "textarea", required: true, source: "text" },
  ],
});

registerPage({
  key: "ml/transcribe",
  title: "Audio Transcription",
  category: "audio",
  path: "/ml/transcribe",
  method: "POST",
  isUpload: true,
  resultKind: "json",
  // Video is a first-class transcription input: whisper demuxes the audio
  // track via ffmpeg server-side, so an mp4/mkv rides the same pipeline as
  // a bare audio file — no client-side extraction needed.
  accepts: ["audio", "video"],
  produces: "transcribe",
  fields: [
    { name: "file", label: "Audio/video file", kind: "file", required: true, source: "selectedIds" },
  ],
});

registerPage({
  key: "ml/vision",
  title: "Image Analysis",
  category: "image",
  path: "/ml/vision",
  method: "POST",
  isUpload: true,
  resultKind: "json",
  accepts: ["image"],
  produces: "analyze",
  fields: [
    { name: "image_path", label: "Image", kind: "file", required: true, source: "selectedIds" },
    { name: "prompt", label: "Prompt", kind: "textarea", default: "Please describe this image." },
    { name: "max_new_tokens", label: "Max new tokens", kind: "number", default: 512 },
  ],
});

registerPage({
  key: "ml/extract",
  title: "Read Document",
  category: "document",
  path: "/ml/extract",
  method: "POST",
  isUpload: true,
  resultKind: "json",
  accepts: ["pdf", "document"],
  produces: "extract",
  fields: [
    { name: "file", label: "Document", kind: "file", required: true, source: "selectedIds" },
  ],
});

registerPage({
  key: "ml/fetch",
  title: "Read Webpage",
  category: "url",
  path: "/ml/fetch",
  method: "POST",
  resultKind: "json",
  accepts: ["url"],
  produces: "extract",
  // The dispatch builder maps a `url`-sourced field (the composer's detected link;
  // see main.tsx firstUrl) into the POST body; the server fetches + extracts readable
  // text, SSRF-guarded. Backend: /ml/fetch (task url-extraction).
  fields: [
    { name: "url", label: "URL", kind: "text", required: true, source: "url" },
  ],
});

registerPage({
  key: "text/generate",
  title: "Text Generation",
  category: "text",
  // Plain LLM generation rides the generic /prompt verb (task fixed via the
  // hidden field's default; not an /ml amenity).
  path: "/prompt",
  method: "POST",
  resultKind: "json",
  accepts: ["text"],
  produces: "chat",
  fields: [
    { name: "task", label: "Task", kind: "text", default: "text-generation" },
    { name: "prompt", label: "Prompt", kind: "textarea", required: true, source: "text" },
    { name: "max_new_tokens", label: "Max new tokens", kind: "number", default: 100 },
    { name: "temperature", label: "Temperature", kind: "number", default: 0.6 },
    { name: "top_p", label: "Top-p", kind: "number", default: 0.95 },
  ],
});
