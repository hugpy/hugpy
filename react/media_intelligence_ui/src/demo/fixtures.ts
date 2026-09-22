// Canned fixtures for the media-intelligence CANNED demo (?demo=1).
//
// Mirrors the dev UI showroom's fixtures: every hugpy endpoint the arm calls is
// answered from a believable, on-brand sample so the REAL chat shell renders
// fully populated with zero backend. Shapes match the live contracts exactly
// (see api.ts / dispatch.ts / fileUpload.ts / mediaIntelligence.ts and the
// output renderers), so the demo exercises the same code paths as production.

import type { ChatEntry } from "../chat/src/imports";
import type { Conversation } from "../chat/src/utilities/chatHistory";

// ── /api/version (entry.tsx contract guard expects { api: 1 }) ───────────────
export const VERSION = { ok: true, api: 1, version: "demo", service: "hugpy" };

// ── /api/models → chat-capable, media-flagged models (constants.ts toOption) ──
export const MODELS = [
  {
    model_key: "Qwen2.5-3B-Instruct-GGUF",
    name: "Qwen2.5 3B Instruct",
    primary_task: "text-generation",
    media: true,
    media_default: true,   // designated default → floated first + preselected
    status: "installed",
  },
  {
    model_key: "Qwen2.5-7B-Instruct-GGUF",
    name: "Qwen2.5 7B Instruct",
    primary_task: "text-generation",
    media: true,
    status: "installed",
  },
  {
    model_key: "Qwen2-VL-2B-Instruct-GGUF",
    name: "Qwen2-VL 2B (vision)",
    primary_task: "image-text-to-text",
    media: true,
    status: "installed",
  },
];

// ── /api/prompt/tasks (server task catalog; arm rarely reads it, but be safe) ─
export const PROMPT_TASKS = {
  tasks: [
    "text-generation",
    "summarization",
    "automatic-speech-recognition",
    "image-text-to-text",
    "feature-extraction",
  ],
  defaults: { "text-generation": MODELS[0].model_key },
};

// ── /api/ml/gate (media access gate; open in the demo) ───────────────────────
export const ML_GATE = { require_key: false };

// ── /ml/* canned tool outputs (shapes per outputRegistry renderers) ──────────
export const SUMMARY_RESULT = {
  text:
    "The document covers Q3 performance: revenue of $48.2M (up 18% YoY) driven " +
    "by enterprise expansion, gross margin improving to 71%, and net retention " +
    "at 124%. Management guided Q4 revenue to $51–53M and reiterated a path to " +
    "break-even by mid next year.",
};

export const KEYWORDS_RESULT = {
  primary: ["revenue growth", "enterprise expansion", "net retention", "gross margin"],
  secondary: ["Q4 guidance", "break-even", "operating leverage", "free cash flow"],
  hashtags: ["#earnings", "#SaaS", "#growth"],
  meta_keywords: ["q3 earnings", "revenue 48.2M", "124% NRR"],
};

export const EXTRACT_RESULT = {
  title: "Sample document",
  text:
    "This is a sample document read by hugpy's media-intelligence demo. In the " +
    "live product, /ml/extract pulls clean, readable text out of PDFs, Office " +
    "documents, and unknown text formats so the model can reason over them.",
};

export const TRANSCRIBE_RESULT = {
  text:
    "Thanks everyone for joining the Q3 earnings call. We delivered revenue of " +
    "forty-eight point two million dollars, up eighteen percent year over year, " +
    "with net retention of one hundred twenty-four percent.",
  language: "en",
  segments: [
    { start: 0.0, end: 4.1, text: "Thanks everyone for joining the Q3 earnings call." },
    { start: 4.1, end: 9.8, text: "We delivered revenue of forty-eight point two million dollars, up eighteen percent year over year," },
    { start: 9.8, end: 13.2, text: "with net retention of one hundred twenty-four percent." },
  ],
};

export const VISION_RESULT = {
  text:
    "The image shows a line chart of quarterly revenue trending upward across " +
    "four quarters, with the steepest rise in the most recent quarter.",
};

export const EMBED_RESULT = {
  embeddings: [Array.from({ length: 8 }, (_, i) => Number((Math.sin(i + 1) / 2).toFixed(4)))],
  dims: 8,
};

// A 1×1 transparent PNG. ImageGenOutput expects GeneratedImage records carrying
// BARE base64 (it prepends `data:image/png;base64,` itself) — not data-URL
// strings — so the shape here must be { images: [{ b64 }] }.
export const IMAGINE_RESULT = {
  images: [
    {
      b64: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
      width: 1,
      height: 1,
    },
  ],
};

export const FETCH_RESULT = {
  title: "Example Domain",
  url: "https://example.com",
  text:
    "Example Domain. This domain is for use in illustrative examples. In the " +
    "live product /ml/fetch reads a page (SSRF-guarded) and returns clean text " +
    "plus title, description, and on-page links for the model to summarize.",
  description: "Illustrative example page",
};

// ── canned chat replies (streamed token-by-token over /chat/stream) ──────────
// Cycled per typed turn so the demo always answers conversationally.
export const CHAT_REPLIES: string[] = [
  "I'm hugpy's media-intelligence assistant. I can transcribe audio, read documents " +
    "and web pages, describe images, and pull out summaries and keywords — then talk " +
    "through what I found. Drop a file or paste a link to see it work. (This is the " +
    "canned demo, so I'm replaying sample responses rather than running a live model.)",
  "Here's how I'd approach that: I'd extract the underlying content first — transcript " +
    "for audio/video, text for documents, a caption for images — then summarize and tag " +
    "the key topics so you get a structured read, not just a wall of text. Attach " +
    "something and I'll show the full breakdown.",
  "Good question. In the live product every answer is grounded in a real tool run " +
    "against your own hugpy backend; in this demo I'm narrating prepared results so you " +
    "can explore the experience end-to-end without any setup.",
];

// ── seeded worked example (the earnings-call thread, shown on load in canned) ─
const SEED_CONV_ID = "demo-earnings-call";
const SEED_AUDIO = "q3-earnings-call.mp3";
const T0 = "2026-06-20T15:03:40.000Z";
const T1 = "2026-06-20T15:03:52.000Z";

const SEED_NARRATION =
  "Here's the gist of the call: it's the Q3 earnings update, and the headline is a " +
  "strong quarter. Revenue came in at $48.2M — up 18% year over year — with net " +
  "retention at 124% and gross margin improving to 71%. Management guided Q4 to " +
  "$51–53M and reiterated a path to break-even by mid next year. I've pulled the full " +
  "transcript, a summary, and the key topics below.";

function seedDocumentIntelligence(): Record<string, unknown> {
  return {
    source: SEED_AUDIO,
    kind: "audio",
    ok: true,
    text: TRANSCRIBE_RESULT.text,
    transcript: TRANSCRIBE_RESULT,
    summary: SUMMARY_RESULT.text,
    keywords: KEYWORDS_RESULT,
    stages: [
      { stage: "transcribe", status: "ok" },
      { stage: "summarize", status: "ok" },
      { stage: "keywords", status: "ok" },
    ],
  };
}

export function seedChats(): ChatEntry[] {
  const model = MODELS[0].model_key;
  return [
    {
      kind: "chat",
      id: "demo-turn-user",
      query: "Summarize this earnings-call recording and pull the key numbers.",
      response: SEED_NARRATION,
      status: "complete",
      finishReason: "stop",
      toolUsed: "Document intelligence",
      files: [{ id: "demo-q3", name: SEED_AUDIO, kind: "audio" }],
      modelKey: model,
      modelLabel: model,
      queryStartedAt: T0,
      responseFinishedAt: T1,
    },
    {
      kind: "tool",
      id: "demo-turn-tool",
      specKey: "media/intelligence",
      title: `Document intelligence · ${SEED_AUDIO}`,
      operation: "intelligence",
      result: seedDocumentIntelligence(),
      status: "complete",
      queryStartedAt: T0,
      responseFinishedAt: T1,
    },
  ];
}

// Build a DocumentIntelligence record for an uploaded file of `kind` — the
// canned answer for POST /api/media/analyze (the chat's attachment path), so a
// visitor who drops their own file still gets a full structured panel.
export function cannedDI(kind: string, source: string): Record<string, unknown> {
  const base = { source: source || "your-file", kind, ok: true } as Record<string, unknown>;
  if (kind === "audio" || kind === "video") {
    return {
      ...base,
      text: TRANSCRIBE_RESULT.text,
      transcript: TRANSCRIBE_RESULT,
      summary: SUMMARY_RESULT.text,
      keywords: KEYWORDS_RESULT,
      stages: [
        { stage: "transcribe", status: "ok" },
        { stage: "summarize", status: "ok" },
        { stage: "keywords", status: "ok" },
      ],
    };
  }
  if (kind === "image") {
    return {
      ...base,
      caption: VISION_RESULT.text,
      text: VISION_RESULT.text,
      keywords: KEYWORDS_RESULT,
      stages: [
        { stage: "caption", status: "ok" },
        { stage: "keywords", status: "ok" },
      ],
    };
  }
  // pdf / document / text
  return {
    ...base,
    text: EXTRACT_RESULT.text,
    summary: SUMMARY_RESULT.text,
    keywords: KEYWORDS_RESULT,
    stages: [
      { stage: "extract", status: "ok" },
      { stage: "summarize", status: "ok" },
      { stage: "keywords", status: "ok" },
    ],
  };
}

export function seedConversationId(): string {
  return SEED_CONV_ID;
}

export function seedConversations(): Conversation[] {
  return [
    {
      id: SEED_CONV_ID,
      title: "Summarize this earnings-call recording…",
      updatedAt: T1,
      messages: seedChats(),
    },
  ];
}
