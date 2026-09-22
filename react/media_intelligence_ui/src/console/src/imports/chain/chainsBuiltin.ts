// Built-in multi-step pipelines.
//
// Importing pagesBuiltin first guarantees the /ml step pages exist before
// registerChain validates + projects each chain. Every chain projects into the
// page registry as a `chain:<key>` PageSpec, so it shows up in BOTH the Tool
// Console picker and the chat tool tray automatically — no per-surface wiring.
//
// Each step's produced media-kind must be accepted by the next step (validated
// at registration). All three below resolve to text → text → summary.
import "../pages/pagesBuiltin"; // step pages must be registered first
import { registerChain } from "./chainRegistry";

// Document → readable text → concise summary.
registerChain({
  key: "doc.brief",
  title: "Document Brief",
  category: "chains",
  description: "Extract a document's text, then summarize it.",
  steps: [{ pageKey: "ml/extract" }, { pageKey: "ml/summarize" }],
});

// Webpage → readable text → concise summary.
registerChain({
  key: "web.brief",
  title: "Webpage Brief",
  category: "chains",
  description: "Read a webpage, then summarize it.",
  steps: [{ pageKey: "ml/fetch" }, { pageKey: "ml/summarize" }],
});

// Audio → transcript → concise summary.
registerChain({
  key: "audio.notes",
  title: "Audio Notes",
  category: "chains",
  description: "Transcribe audio, then summarize the transcript.",
  steps: [{ pageKey: "ml/transcribe" }, { pageKey: "ml/summarize" }],
});
